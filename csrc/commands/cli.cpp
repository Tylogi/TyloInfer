#include "cli.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace tyloi::commands {

std::vector<std::string> split_csv_values(
        const std::string& value,
        const char* option) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        const auto last = item.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            throw std::runtime_error(
                std::string(option) + " contains an empty item");
        }
        result.push_back(item.substr(first, last - first + 1));
    }
    if (result.empty()) {
        throw std::runtime_error(
            std::string(option) + " requires at least one value");
    }
    if (value.ends_with(',')) {
        throw std::runtime_error(
            std::string(option) + " contains an empty item");
    }
    return result;
}

int with_command_errors(const std::function<int()>& fn) {
    try {
        return fn();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
namespace {

int strict_int(const std::string& text, const char* option) {
    int result = 0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string(option) + " requires integers");
    }
    return result;
}

double strict_double(const std::string& text, const char* option) {
    double result = 0.0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), result,
        std::chars_format::general);
    if (error != std::errc{} || end != text.data() + text.size() ||
            !std::isfinite(result)) {
        usage_error(std::string(option) + " requires finite numbers");
    }
    return result;
}

std::vector<std::string> command_csv_values(
        const std::string& value, const char* option) {
    try {
        return split_csv_values(value, option);
    } catch (const std::runtime_error& error) {
        usage_error(error.what());
    }
}

std::vector<int> parse_parallel_devices(
        const std::string& value, const char* option) {
    const auto values = command_csv_values(value, option);
    std::vector<int> devices;
    if (values.size() == 1 &&
            value.find(',') == std::string::npos) {
        const int count = strict_int(values.front(), option);
        if (count < 2) {
            usage_error(std::string(option) +
                " device count must be at least two");
        }
        devices.resize(static_cast<size_t>(count));
        std::iota(devices.begin(), devices.end(), 0);
    } else {
        for (const auto& item : values) {
            devices.push_back(strict_int(item, option));
        }
        if (devices.size() < 2) {
            usage_error(std::string(option) +
                " requires at least two devices");
        }
    }
    return devices;
}

std::vector<double> parse_parallel_split(
        const std::string& value, const char* option) {
    std::vector<double> split;
    for (const auto& item : command_csv_values(value, option)) {
        const double weight = strict_double(item, option);
        if (!std::isfinite(weight) || weight <= 0.0) {
            usage_error(std::string(option) +
                " values must be finite and positive");
        }
        split.push_back(weight);
    }
    return split;
}

std::unordered_set<int> parse_layer_ranges(const std::string& value) {
    std::unordered_set<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(
            std::remove_if(
                item.begin(), item.end(),
                [](unsigned char ch) { return std::isspace(ch) != 0; }),
            item.end());
        if (item.empty()) {
            usage_error("--cpu-offload-layers contains an empty item");
        }
        const size_t dash = item.find('-');
        int first = 0;
        int last = 0;
        if (dash == std::string::npos) {
            first = last = strict_int(item, "--cpu-offload-layers");
        } else {
            if (dash == 0 || dash + 1 >= item.size() ||
                item.find('-', dash + 1) != std::string::npos) {
                usage_error("invalid --cpu-offload-layers range: " + item);
            }
            first = strict_int(
                item.substr(0, dash), "--cpu-offload-layers");
            last = strict_int(
                item.substr(dash + 1), "--cpu-offload-layers");
        }
        if (first < 0 || last < first) {
            usage_error("invalid --cpu-offload-layers range: " + item);
        }
        for (int layer = first; layer <= last; ++layer) {
            result.insert(layer);
        }
    }
    if (result.empty()) {
        usage_error("--cpu-offload-layers requires at least one layer");
    }
    return result;
}

} // namespace

bool parse_load_option(
        std::string_view option,
        ArgCursor& args,
        tyloi::engine::LoadOptions& result) {
    if (option == "--model") result.model_path = args.value(option);
    else if (option == "--config") result.config_path = args.value(option);
    else if (option == "--yarn-context-extension") result.yarn_context_extension = true;
    else if (option == "--threads" || option == "-t") {
        result.cpu_threads = integer<int>(args.value(option), option);
        if (result.cpu_threads <= 0) usage_error("--threads must be positive");
        result.cpu_threads_set = true;
    }
    else if (option == "--ctx-size") {
        result.context_size = integer<int64_t>(args.value(option), option);
        if (result.context_size < 0) {
            usage_error("--ctx-size must be non-negative");
        }
    }
    else if (option == "--cpu-offload-layers") {
        result.cpu_offload_layers = parse_layer_ranges(args.value(option));
    }
    else if (option == "--n-gpu-layers" || option == "-ngl") {
        result.n_gpu_layers = integer<int>(args.value(option), option);
        if (result.n_gpu_layers < 0) {
            usage_error("--n-gpu-layers must be non-negative");
        }
        result.n_gpu_layers_set = true;
    }
    else if (option == "--moe-gpu-cache-gb") {
        result.moe_gpu_cache_gb = number(args.value(option), option);
        if (result.moe_gpu_cache_gb < 0.0) {
            usage_error("--moe-gpu-cache-gb must be non-negative");
        }
    }
    else if (option == "--moe-cache-profile") {
        result.moe_cache_profile_path = args.value(option);
    }
    else if (option == "--dflash2-draft") {
        result.dflash2_draft_path = args.value(option);
    }
    else if (option == "--tensor-parallel") {
        result.tensor_parallel.devices = parse_parallel_devices(
            args.value(option), "--tensor-parallel");
    }
    else if (option == "--tensor-split") {
        result.tensor_parallel.split = parse_parallel_split(
            args.value(option), "--tensor-split");
    }
    else if (option == "--expert-parallel") {
        result.expert_parallel.devices = parse_parallel_devices(
            args.value(option), "--expert-parallel");
    }
    else if (option == "--expert-split") {
        result.expert_parallel.split = parse_parallel_split(
            args.value(option), "--expert-split");
    }
    else if (option == "--layer-parallel") {
        result.layer_placement.devices = parse_parallel_devices(
            args.value(option), "--layer-parallel");
    }
    else if (option == "--layer-split") {
        result.layer_placement.split = parse_parallel_split(
            args.value(option), "--layer-split");
    }
    else if (option == "--parallel-test-duplicates" ||
            option == "--tensor-parallel-test-duplicates") {
        result.tensor_parallel.allow_duplicate_devices = true;
        result.expert_parallel.allow_duplicate_devices = true;
    }
    else if (option == "--tokenizer") {
        result.tokenizer_model = args.value(option);
    }
    else return false;
    return true;
}

} // namespace tyloi::commands
