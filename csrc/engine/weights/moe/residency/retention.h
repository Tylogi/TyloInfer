#pragma once

#include "engine/weights/moe/residency/heat.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

namespace tyloi::weights {

struct ExpertRetention {
    std::uint64_t key;
    int device;
    std::size_t bytes;
    float heat;
    bool retained;
};

// Items belong to one budget and have finite heat. Only retained flags change.
inline void select_expert_retention(std::span<ExpertRetention> items, std::size_t budget, bool rebuild) {
    const auto hotter = [](const ExpertRetention* a, const ExpertRetention* b) {
        return a->heat != b->heat ? a->heat > b->heat :
            std::tie(a->key, a->device) < std::tie(b->key, b->device);
    };
    std::vector<ExpertRetention*> candidates, victims;
    std::size_t used = 0;
    for (auto& item : items) {
        if (item.retained) {
            if (item.bytes > budget - used) rebuild = true;
            else used += item.bytes;
            victims.push_back(&item);
        } else if (item.heat >= 2.f) {
            candidates.push_back(&item);
        }
    }

    if (rebuild) {
        candidates.clear();
        for (auto& item : items) candidates.push_back(&item);
        std::stable_sort(candidates.begin(), candidates.end(), hotter);
        used = 0;
        for (auto* item : candidates) {
            item->retained = item->heat >= engine::expert_replacement_margin && item->bytes <= budget - used;
            if (item->retained) used += item->bytes;
        }
        return;
    }

    std::stable_sort(candidates.begin(), candidates.end(), hotter);
    std::stable_sort(victims.begin(), victims.end(), [&](const auto* a, const auto* b) { return hotter(b, a); });
    std::size_t first = 0, admitted = 0;
    for (auto* candidate : candidates) {
        if (admitted == 96) break;
        std::size_t available = budget - used, end = first;
        float heat = 0;
        while (end < victims.size() && available < candidate->bytes) {
            heat += victims[end]->heat;
            available += victims[end]->bytes;
            ++end;
        }
        if (available < candidate->bytes || candidate->heat < heat + engine::expert_replacement_margin) continue;
        while (first < end) {
            victims[first]->retained = false;
            used -= victims[first++]->bytes;
        }
        candidate->retained = true;
        used += candidate->bytes;
        assert(used <= budget);
        ++admitted;
    }
}

} // namespace tyloi::weights
