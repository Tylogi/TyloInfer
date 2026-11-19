#pragma once

#include "engine/models/session/paged.h"

struct TextSessionState;

namespace tyloi::models::qwen4_exp {

SessionBlockPayload encode_block(const TextSessionState&, std::size_t, std::size_t);
TextSessionState decode_session(const std::vector<SessionBlockPayload>&,
                                const std::vector<int64_t>&, std::size_t);
bool exact(const SessionBlockPayload&);
bool checkpoint_needs_update(const SessionBlockPayload&, const TextSessionState&);

} // namespace tyloi::models::qwen4_exp
