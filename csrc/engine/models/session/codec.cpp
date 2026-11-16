#include "engine/models/session/codec.h"
#include "engine/models/session/cache.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/deepseek_v41/session_codec.h"
#include "engine/models/qwen4_exp/session_codec.h"
#include "engine/models/source/model.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace tyloi::models {

std::string prefix_cache_compatibility_key(
    const tyloi::ModelSource &source, int64_t max_position_embeddings,
    tyloi::Device device, std::string_view kv_fingerprint) {
    if (!device.is_cuda() && !device.is_metal())
        throw std::invalid_argument("prefix cache requires a supported device");
    std::ostringstream key;
    const auto backend = device.is_cuda() ? "cuda" : "metal";
    key << "mfq-" << backend << "-prefix-v1\n"
        << "codec=" << backend << '-' << (source.architecture()=="qwen4_exp" ? "qwen4-session-v1" :
            source.architecture()=="deepseek_v41" ? "deepseek-v41-session-v1" : "full-attention-kv-v1") << '\n'
        << "context=" << max_position_embeddings << '\n'
        << "architecture=" << source.architecture() << '\n';
    if(!kv_fingerprint.empty())key<<"kv="<<kv_fingerprint<<'\n';
    for (std::size_t index = 0; index < source.source_paths().size(); ++index) {
        const auto &path = source.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source for prefix cache: " + error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error("cannot identify model source timestamp: " + error.message());
        }
        const auto modified_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(modified.time_since_epoch())
                .count();
        key << "source=" << index << ':' << path.filename().string() << ':' << size << ':'
            << modified_ns << '\n';
    }
    for (const auto &tensor : source.tensors()) {
        key << "tensor=" << tensor.name << ':' << tensor.dtype << ':' << tensor.nbytes << '\n';
    }
    return key.str();
}

TextSessionState SessionOps::decode(
    const std::vector<SessionBlockPayload>& payloads,
    const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    if (!payloads.empty() && payloads.front() && payloads.front()->size() > 6 && (*payloads.front())[6] == '2')
        return tyloi::models::qwen4_exp::decode_session(payloads, tokens, block_size);
    if (!payloads.empty() && payloads.front() && payloads.front()->size() > 6 && (*payloads.front())[6] == '3')
        return tyloi::models::deepseek_v41::decode_session(payloads, tokens, block_size);
    return decode_paged_session(payloads, tokens, block_size);
}

SessionBlockPayload SessionOps::encode(
    const Snapshot& state, std::size_t block_size, std::size_t index) {
    if (state.kind() == TextSessionStateKind::Qwen4)
        return tyloi::models::qwen4_exp::encode_block(state, block_size, index);
    if (state.kind() == TextSessionStateKind::DeepseekV41)
        return tyloi::models::deepseek_v41::encode_block(state, block_size, index);
    return encode_paged_block(state, block_size, index);
}

bool SessionOps::paged_mtp(const Snapshot& state) {
    return state.kind() == TextSessionStateKind::Qwen4 ||
        state.kind() == TextSessionStateKind::DeepseekV41;
}

bool SessionOps::partial_blocks(const Snapshot& state) {
    return paged_mtp(state);
}

bool SessionOps::paged_media(const Snapshot& state) {
    return state.kind() == TextSessionStateKind::DeepseekV41;
}

bool SessionOps::checkpoint_needs_update(
    const SessionBlockPayload& payload, const Snapshot& state) {
    if (state.kind() != TextSessionStateKind::DeepseekV41) return false;
    return tyloi::models::deepseek_v41::checkpoint_needs_update(payload, state);
}

std::size_t SessionOps::decodable_blocks(
    const std::vector<SessionBlockPayload>& payloads) {
    return paged_decodable_blocks(payloads);
}

std::size_t SessionOps::token_count(const SessionBlockPayload& payload) {
    return paged_payload_tokens(payload);
}

} // namespace tyloi::models

size_t paged_payload_tokens(const SessionBlockPayload& payload) {
    if(!payload)throw CudaSessionStateError("null CUDA prefix block");
    SessionBlockReader reader(*payload);std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    constexpr std::array<uint8_t,8> qwen4_magic{'M','F','Q','C','U','D','2',0};
    constexpr std::array<uint8_t,8> full_magic{'M','F','Q','C','U','D','1',0};
    constexpr std::array<uint8_t,8> deepseek_magic{'M','F','Q','C','U','D','3',0};
    const auto version=reader.scalar<uint32_t>("version");
    if((magic!=qwen4_magic && magic!=full_magic && magic!=deepseek_magic) ||
        (version!=1 && !(magic==qwen4_magic && version==2)))
        throw CudaSessionStateError("invalid CUDA prefix block header");
    (void)reader.scalar<uint32_t>("start");return reader.scalar<uint32_t>("count");
}
size_t paged_decodable_blocks(const std::vector<SessionBlockPayload>& payloads) {
    for(size_t count=payloads.size();count>0;--count) {
        const auto& payload=payloads[count-1];
        (void)paged_payload_tokens(payload);
        if((*payload)[6]=='1')return count;
        if((*payload)[6]=='3') {
            if(tyloi::models::deepseek_v41::exact(payload))return count;
            continue;
        }
        if(tyloi::models::qwen4_exp::exact(payload))return count;
    }
    return 0;
}
