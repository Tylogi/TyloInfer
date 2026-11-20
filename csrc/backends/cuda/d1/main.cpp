#include "engine.h"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        std::filesystem::path assets, weights;
        bool fused = true, stream = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--unfused") fused = false;
            else if (arg == "--weights-placement" && i + 1 < argc) {
                const std::string value = argv[++i];
                if (value != "stream" && value != "vram") throw std::invalid_argument("weights placement must be stream or vram");
                stream = value == "stream";
            }
            else if ((arg == "--assets" || arg == "--weights") && i + 1 < argc) {
                auto value = std::filesystem::u8path(argv[++i]);
                if (arg == "--assets") assets = std::move(value); else weights = std::move(value);
            } else throw std::invalid_argument("usage: tyloinfer-d1-native --assets <HF-directory> [--weights <HF-directory-or-GGUF>] [--weights-placement stream|vram] [--unfused]");
        }
        if (assets.empty()) throw std::invalid_argument("d1 native requires local configuration assets");
        if (weights.empty()) weights = assets;
        mfq::cuda::d1::Input input(assets, weights);
        mfq::cuda::d1::Engine engine(input, fused, stream);
        std::cout << engine.ready().dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto request = nlohmann::json::parse(line);
                std::cout << engine.evaluate(request).dump() << std::endl;
            } catch (const std::exception& error) {
                std::cout << nlohmann::json({{"error", error.what()}}).dump() << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "d1-native: " << error.what() << std::endl;
        return 2;
    }
}
