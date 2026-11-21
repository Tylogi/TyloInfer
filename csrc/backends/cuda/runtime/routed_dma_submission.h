#pragma once

namespace mfq::cuda {
// A GPU consumer is waiting for this host submission. Its producer lane
// must make progress independently of the consumer's compute scheduling.
class RoutedHostSubmissionScope {
    inline static thread_local unsigned depth_=0;
public:
    RoutedHostSubmissionScope(){++depth_;}
    RoutedHostSubmissionScope(const RoutedHostSubmissionScope&)=delete;
    RoutedHostSubmissionScope& operator=(const RoutedHostSubmissionScope&)=delete;
    ~RoutedHostSubmissionScope(){--depth_;}
    static bool active()noexcept{return depth_!=0;}
};

inline unsigned routed_native_copy_flags(int mode)noexcept {
    // CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE is optional. Do not request
    // compute overlap when that compute depends on the copies being served.
    return mode>=2 && !RoutedHostSubmissionScope::active()?1u:0u;
}
} // namespace mfq::cuda
