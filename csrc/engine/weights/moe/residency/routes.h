#pragma once

#include "heat.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyloi::engine {

class RouteHistory {
public:
    // CUDA packed MTP admits at most 32 requests times five draft rows plus
    // the target row.  The mask API keeps each of those physical rows
    // independently, including holes between accepted requests.
    static constexpr int max_speculative_rows = 32 * (5 + 1);
    enum class HeatMode { count, unique, normalized, executed };
    struct Bundle { int experts; bool verified; };
    struct Route { std::size_t bundle; int32_t expert; };
    struct Stats {
        std::uint64_t observed = 0, windows = 0, committed = 0, discarded = 0,
                      cancels = 0, heat_routes = 0;
    };
    struct Accepted { std::uint64_t routes = 0, heat_routes = 0; };

    RouteHistory(const std::vector<Bundle>& layout, HeatMode mode, float retention)
        : mode_(mode), retention_(retention) {
        const auto mode_value = static_cast<int>(mode);
        if (mode_value < static_cast<int>(HeatMode::count) ||
            mode_value > static_cast<int>(HeatMode::executed) ||
            !std::isfinite(retention) || retention < 0 || retention > 1)
            throw std::invalid_argument("invalid expert heat policy");
        histories_.reserve(layout.size());
        for (const auto& bundle : layout) {
            if (bundle.experts <= 0)
                throw std::invalid_argument("invalid expert history layout");
            histories_.push_back({bundle.verified,
                std::vector<float>(std::size_t(bundle.experts)),
                std::vector<float>(std::size_t(bundle.experts)), {},
                std::vector<std::uint64_t>((std::size_t(bundle.experts) + 63) / 64),
                {}});
        }
    }

    void begin(int rows) {
        if (rows < 1 || rows > max_speculative_rows)
            throw std::invalid_argument("speculative route row count is out of range");
        if (pending_rows_)
            throw std::logic_error("speculative routes already pending");
        for (auto& history : histories_) {
            history.ids.clear();
            history.rows = 0;
            history.routes = 0;
            history.published.clear();
            history.learning.clear();
        }
        pending_rows_ = rows;
    }
    bool pending() const noexcept { return pending_rows_ != 0; }

    void observe(std::size_t bundle, std::span<const int32_t> ids, int rows,
                 bool independent_sequences = false, int row_offset = 0,
                 int row_total = 0, bool learning = true) {
        auto& history = histories_.at(bundle);
        const bool deferred = pending() && history.verified;
        if (deferred && (independent_sequences || rows < 1 || rows > pending_rows_ ||
                         ids.size() % std::size_t(rows)))
            throw std::logic_error("speculative routes do not describe one aligned sequence");
        validate_experts(history, ids);
        if (deferred) {
            if (row_total == 0) row_total = pending_rows_;
            if (row_offset < 0 || row_total != pending_rows_ ||
                row_offset > row_total || rows > row_total - row_offset)
                throw std::logic_error("speculative route row coordinates are not aligned");
            const auto routes = ids.size() / std::size_t(rows);
            if (routes == 0 || routes > std::size_t(std::numeric_limits<int>::max()))
                throw std::logic_error("speculative route width is out of range");
            if (!history.rows) {
                history.rows = row_total;
                history.routes = static_cast<int>(routes);
                history.ids.assign(std::size_t(row_total) * routes, -1);
                history.published.assign(std::size_t(row_total), std::uint8_t(0));
                history.learning.assign(std::size_t(row_total), std::uint8_t(0));
            } else if (history.rows != row_total || history.routes != static_cast<int>(routes)) {
                throw std::logic_error("speculative route publication geometry changed");
            }
            std::copy(ids.begin(), ids.end(), history.ids.begin() +
                std::size_t(row_offset) * routes);
            std::fill(history.published.begin() + row_offset,
                      history.published.begin() + row_offset + rows, std::uint8_t(1));
            std::fill(history.learning.begin() + row_offset,
                      history.learning.begin() + row_offset + rows, std::uint8_t(learning));
        } else if (learning) {
            for (const auto expert : ids)
                history.observations[expert] += 1.0f;
            observed_ += ids.size();
        }
    }

    // The returned view remains valid until the next selection query.
    std::span<const Route> selected(std::span<const std::uint8_t> keep) {
        validate_keep(keep);
        accepted_.clear();
        if (!pending())
            return accepted_;
        for (int row = 0; row < pending_rows_; ++row) {
            if (!keep[std::size_t(row)]) continue;
            for (std::size_t bundle = 0; bundle < histories_.size(); ++bundle) {
                const auto& history = histories_[bundle];
                if (!history.rows || row >= history.rows ||
                    !history.published[std::size_t(row)] || !history.learning[std::size_t(row)])
                    continue;
                for (int route = 0; route < history.routes; ++route)
                    accepted_.push_back({bundle, history.ids[std::size_t(row) * history.routes + route]});
            }
        }
        return accepted_;
    }

    Accepted commit(std::span<const std::uint8_t> keep, bool executed = false, bool learning = true) {
        validate_keep(keep);
        Accepted result;
        if (!pending())
            return result;
        const bool accepted = std::any_of(keep.begin(), keep.end(), [](std::uint8_t value) { return value != 0; });
        const bool all_rows = accepted && executed && mode_ == HeatMode::executed;
        const bool unique = mode_ == HeatMode::unique || mode_ == HeatMode::normalized;
        for (auto& history : histories_) {
            if (!history.rows)
                continue;
            std::size_t count = 0, committed = 0;
            if (unique)
                std::fill(history.seen.begin(), history.seen.end(), std::uint64_t(0));
            std::size_t distinct = 0;
            for (int row = 0; row < pending_rows_; ++row) {
                if (row >= history.rows || !history.published[std::size_t(row)]) continue;
                if (keep[std::size_t(row)]) committed += std::size_t(history.routes);
                if (!learning || !history.learning[std::size_t(row)] ||
                    (!all_rows && !keep[std::size_t(row)])) continue;
                for (int route = 0; route < history.routes; ++route) {
                    const auto expert = unsigned(history.ids[std::size_t(row) * history.routes + route]);
                    ++count;
                    if (unique) {
                        auto& word = history.seen[expert / 64];
                        const auto bit = std::uint64_t(1) << (expert % 64);
                        if (word & bit)
                            continue;
                        word |= bit;
                    }
                    ++distinct;
                }
            }
            // Preserve total token heat while valuing each saved transfer equally.
            const float weight = mode_ == HeatMode::normalized && distinct
                ? float(count) / float(distinct) : 1.0f;
            for (int row = 0; row < pending_rows_; ++row) {
                if (row >= history.rows || !history.published[std::size_t(row)] ||
                    !learning || !history.learning[std::size_t(row)] ||
                    (!all_rows && !keep[std::size_t(row)])) continue;
                for (int route = 0; route < history.routes; ++route) {
                    const auto expert = unsigned(history.ids[std::size_t(row) * history.routes + route]);
                    if (unique) {
                        auto& word = history.seen[expert / 64];
                        const auto bit = std::uint64_t(1) << (expert % 64);
                        if (!(word & bit))
                            continue;
                        word &= ~bit;
                    }
                    history.observations[expert] += weight;
                }
            }
            result.routes += committed;
            result.heat_routes += distinct;
            discarded_ += std::size_t(std::count(history.published.begin(), history.published.end(),
                                                 std::uint8_t(1))) * history.routes - committed;
            observed_ += count;
        }
        committed_ += result.routes;
        heat_routes_ += result.heat_routes;
        ++windows_;
        pending_rows_ = 0;
        return result;
    }

    void discard() noexcept {
        if (!pending())
            return;
        for (auto& history : histories_) {
            discarded_ += std::size_t(std::count(history.published.begin(), history.published.end(),
                                                 std::uint8_t(1))) * history.routes;
            history.ids.clear();
            history.rows = 0;
            history.routes = 0;
            history.published.clear();
            history.learning.clear();
        }
        ++cancels_;
        pending_rows_ = 0;
    }

    void prefill(std::size_t bundle, std::span<const int32_t> ids, int rows, int routes) {
        auto& history = histories_.at(bundle);
        if (rows < 1 || routes < 1 || ids.size() != std::size_t(rows) * routes)
            throw std::invalid_argument("invalid prefill route geometry");
        validate_experts(history, ids);
        prefill_values(history.heat, ids, rows);
        std::fill(history.observations.begin(), history.observations.end(), 0.0f);
        history.prefill_reset = false;
    }

    void stage_prefill(std::size_t bundle, std::span<const int32_t> ids, int rows) {
        auto& history = histories_.at(bundle);
        if (rows < 1 || ids.empty() || ids.size() % std::size_t(rows))
            throw std::invalid_argument("invalid prefill route geometry");
        validate_experts(history, ids);
        prefill_values(history.observations, ids, rows);
        history.prefill_reset = true;
        observed_ += ids.size();
    }

    void flush() {
        for (auto& history : histories_) {
            if (history.prefill_reset) {
                std::fill(history.heat.begin(), history.heat.end(), 0.0f);
                history.prefill_reset = false;
            }
            for (std::size_t expert = 0; expert < history.heat.size(); ++expert) {
                history.heat[expert] += history.observations[expert];
                history.observations[expert] = 0;
            }
        }
    }
    void decay() { decay(retention_); }
    void decay(float retention) {
        for (auto& history : histories_)
            for (float& heat : history.heat)
                heat *= retention;
    }
    float heat(std::size_t bundle, int expert) const {
        return histories_.at(bundle).heat.at(std::size_t(expert));
    }
    float total_heat(std::size_t bundle, int expert) const {
        const auto& history = histories_.at(bundle);
        return history.heat.at(std::size_t(expert)) + history.observations.at(std::size_t(expert));
    }
    Stats stats() const noexcept {
        return {observed_.load(), windows_.load(), committed_.load(), discarded_.load(),
                cancels_.load(), heat_routes_.load()};
    }

private:
    struct History {
        bool verified;
        std::vector<float> heat, observations;
        std::vector<int32_t> ids;
        std::vector<std::uint64_t> seen;
        std::vector<std::uint8_t> published, learning;
        int rows = 0, routes = 0;
        bool prefill_reset = false;
    };

    static void prefill_values(std::vector<float>& values, std::span<const int32_t> ids, int rows) {
        std::vector<std::uint64_t> counts(values.size());
        for (const auto expert : ids)
            ++counts[std::size_t(expert)];
        for (std::size_t expert = 0; expert < counts.size(); ++expert)
            values[expert] = expert_prefill_usage(counts[expert], std::size_t(rows));
    }

    static void validate_experts(const History& history, std::span<const int32_t> ids) {
        for (const auto expert : ids)
            if (unsigned(expert) >= history.heat.size())
                throw std::out_of_range("expert route is out of range");
    }

    void validate_keep(std::span<const std::uint8_t> keep) const {
        if (!pending()) {
            if (!keep.empty())
                throw std::invalid_argument("speculative route mask has no pending window");
            return;
        }
        if (keep.size() != std::size_t(pending_rows_))
            throw std::invalid_argument("speculative route mask row count is out of range");
        if (std::any_of(keep.begin(), keep.end(), [](std::uint8_t value) { return value > 1; }))
            throw std::invalid_argument("speculative route mask contains a non-binary row");
        for (const auto& history : histories_)
            if (history.rows)
                for (std::size_t row = std::size_t(history.rows); row < keep.size(); ++row)
                    if (keep[row])
                        throw std::logic_error("speculative route publication lost row alignment");
        for (const auto& history : histories_)
            if (history.rows && !history.published.empty())
                for (std::size_t row = 0; row < keep.size(); ++row)
                    if (keep[row] && !history.published[row])
                        throw std::logic_error("speculative route publication lost row alignment");
    }

    std::vector<History> histories_;
    std::vector<Route> accepted_;
    HeatMode mode_;
    float retention_;
    int pending_rows_ = 0;
    std::atomic<std::uint64_t> observed_{0}, windows_{0}, committed_{0}, discarded_{0},
        cancels_{0}, heat_routes_{0};
};

} // namespace tyloi::engine
