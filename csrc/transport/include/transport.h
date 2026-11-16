#pragma once

#include "runtime_profile.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct MfqHttpRuntimeTransportConfig : MfqRuntimeTransportConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
};

class MfqTransport {
public:
    virtual ~MfqTransport() = default;
    virtual int run(const tyloi::engine::Engine & engine) = 0;
};

std::unique_ptr<MfqTransport> make_mfq_http_transport(
    MfqHttpRuntimeTransportConfig config);

// Reserve the original stdout pipe for NDJSON and redirect ordinary process
// output to stderr. Call before loading a model so startup logs cannot corrupt
// the protocol stream.
void prepare_mfq_stdio_transport();
std::unique_ptr<MfqTransport> make_mfq_stdio_transport(
    MfqRuntimeTransportConfig config);
