#pragma once

#include "engine/requests/result.h"
#include "engine/tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct TextSessionState;

namespace tyloi { class Executor; }
namespace tyloi::weights { class Cache; }
namespace tyloi::engine { class Budget; }
namespace tyloi::models { class State; struct Mtp; }

namespace tyloi::state {

struct SessionRestore {
    std::size_t tokens = 0;
    Tensor mtp_last_target_hidden;
};

class Sessions {
public:
    virtual ~Sessions() = default;
    virtual bool persistent_prefix_enabled() const noexcept = 0;
    virtual SessionRestore restore_best(models::State& state, Executor& executor,
        weights::Cache& weights, engine::Budget& budget,
        models::Mtp* mtp, const std::string& session, const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix, const std::string& input_key = {}) = 0;
    virtual void store(const std::string& session, ::TextSessionState snapshot) = 0;
    virtual engine::SessionResult control(const engine::SessionCommand& command) = 0;
    virtual std::vector<std::pair<std::string, double>> metrics() const = 0;
    virtual std::map<int, std::size_t> device_hot_bytes() const = 0;
    virtual std::uint64_t trim_hot(std::uint64_t bytes) = 0;
    virtual void reserve_ram(std::uint64_t bytes) = 0;
};

Tensor restore(models::State& state, const TextSessionState& snapshot, Executor& executor,
    weights::Cache& weights, engine::Budget& budget, models::Mtp* mtp);

} // namespace tyloi::state
