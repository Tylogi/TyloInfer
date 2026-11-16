#include "engine/backends/cuda/executor/executor.h"
#include "engine/weights/encoding/tq_table.h"
#include "engine/weights/tq/cache.h"
#include "engine/weights/tq/image.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using Bytes = std::vector<std::uint8_t>;
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
double metric(TqCache& cache, const char* name) {
    for (const auto& [key, value] : cache.metrics()) if (key == name) return value;
    throw std::runtime_error("missing cache metric");
}
Bytes expert(unsigned salt) {
    constexpr unsigned rows = 256, width = 256, gs = 16;
    const unsigned bits = 3 + salt % 2;
    constexpr unsigned scales = rows * (width / gs) / 2;
    const unsigned packed = rows * width * bits / 8;
    Bytes wire(64 + rows * 8 + scales + packed);
    std::memcpy(wire.data(), "RTRELL05", 8);
    for (const auto [offset, value] : std::vector<std::pair<unsigned, unsigned>>{
            {8, 5}, {12, rows}, {16, width}, {24, gs}, {28, packed}})
        tyloi::TqLayout::integer(wire.data() + offset, value, 4);
    for (unsigned row = 0; row < rows; ++row)
        tyloi::TqLayout::integer(wire.data() + 64 + row * 8,
            row * width * bits / 32 | (std::uint64_t(bits - 1) << 32) |
                (std::uint64_t(0x3c00) << 40), 8);
    std::fill(wire.begin() + 64 + rows * 8, wire.end(), std::uint8_t(salt));
    return wire;
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        std::vector<std::shared_ptr<tyloi::TqRows>> rows;
        std::vector<Bytes> expected;
        std::size_t payload = 0;
        for (unsigned i = 0; i < 8; ++i) {
            auto wire = std::make_shared<Bytes>(expert(i));
            payload += wire->size();
            expected.push_back(tq_image(*wire, tyloi::TqLayout(wire->data(), wire->size())));
            rows.push_back(std::make_shared<tyloi::TqRows>(wire->size(),
                [wire](std::uint64_t offset, std::byte* target, std::size_t size) {
                    if (offset > wire->size() || size > wire->size() - offset)
                        throw std::out_of_range("fixture read");
                    std::memcpy(target, wire->data() + offset, size);
                }));
        }
        auto table = std::make_shared<tyloi::TqTable>(std::move(rows), 256, 8 * 256, payload);
        tyloi::cuda::Executor executor({}, 0);
        auto scope = executor.enter();
        TqCache cache(128 << 10, 320 << 10, 0, false, executor.tq());
        const auto id = cache.add(table, 256);
        cache.preload(0, 0);
        cache.prepare_decode(0, 0);
        const auto reads = metric(cache, "ssd_expert_read_count");
        const auto verify = [&](std::int64_t index) {
            auto value = cache.get(id, &index, 1, 0).front().image().cpu();
            check(value.nbytes() == expected[index].size(), "image size changed");
            check(!std::memcmp(value.data_ptr(), expected[index].data(), value.nbytes()), "image bytes changed");
            check(metric(cache, "ssd_expert_read_count") == reads, "resident expert was reread from SSD");
            check(metric(cache, "ssd_expert_payload_bytes") == 0, "exclusive tiers lost an expert");
            check(metric(cache, "resident_expert_cache_bytes") <= cache.budget(), "GPU budget exceeded");
            check(metric(cache, "ram_expert_payload_bytes") <= (320 << 10), "RAM budget exceeded");
        };
        for (int index : {0, 7, 2, 6, 1, 5, 3, 4, 0, 7}) verify(index);
        cache.resize(0);
        check(metric(cache, "resident_expert_cache_bytes") == 0, "GPU shrink retained images");
        for (int index : {0, 7, 2}) verify(index);
        cache.resize(128 << 10);
        cache.prepare_decode(0, 0);
        for (int index : {6, 1, 5}) verify(index);
        check(metric(cache, "expert_cache_evictions") > 0, "test did not exercise eviction");
        executor.drain();
        std::cout << "TQ exclusive GPU/RAM residency, resize and packed-byte preservation passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
