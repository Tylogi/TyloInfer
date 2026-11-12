#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"

namespace tyloi::models {

std::unique_ptr<Restore> MtpState::prepare_restore(const ::MtpSessionState&) {
    throw std::logic_error("predictor session restoration is not implemented");
}

::TextSessionState State::capture_text_session_state(const std::vector<std::int64_t>&) const {
    throw std::logic_error("text session capture is not implemented");
}

std::unique_ptr<Restore> State::prepare_restore(const ::TextSessionState&, const Mtp*) {
    throw std::logic_error("text session restoration is not implemented");
}

} // namespace tyloi::models
