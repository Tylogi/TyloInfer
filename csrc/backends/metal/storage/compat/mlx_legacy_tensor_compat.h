#pragma once

#include "engine/models/source/model_graph.h"

namespace mfq::metal {

class MfqContainer;

// Temporary Metal record-alias boundary. The backend-neutral legacy graph is
// synthesized separately by the shared legacy graph adapter; backend loaders see only
// these canonical aliases. Delete both adapters after the schema-v1 migration
// window.
void install_legacy_tensor_compatibility(MfqContainer& model);

// Every native loader consumes a graph. Canonical artifacts carry it; this
// function synthesizes the one temporary graph for a pre-schema artifact so
// legacy architecture strings never leak into backend dispatch.
tyloi::ModelGraph effective_model_graph(const MfqContainer& model);

} // namespace mfq::metal
