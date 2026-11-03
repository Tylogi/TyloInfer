#pragma once

#include <nlohmann/json.hpp>

namespace mfq::server {
class Storage;

// The three runtime policy documents stored in schema_meta. The JSON in and out mirrors the
// pydantic models in tyloinfer/server/protocol/models.py, so the wire shape needs no extra type.
class RuntimePolicies {
public:
    explicit RuntimePolicies(Storage& storage);
    nlohmann::json context_policy() const;
    nlohmann::json put_context_policy(const nlohmann::json& body);
    nlohmann::json inference_policy() const;
    nlohmann::json put_inference_policy(const nlohmann::json& body);
    nlohmann::json memory_policy() const;
    nlohmann::json put_memory_policy(const nlohmann::json& body);

private:
    Storage& storage_;
};

} // namespace mfq::server
