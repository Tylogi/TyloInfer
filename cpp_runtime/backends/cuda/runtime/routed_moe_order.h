#pragma once

namespace mfq::cuda {
// Resident and streamed outputs occupy disjoint routing slots. Adapters
// supply kernels while this order owns every producer/consumer boundary.
template<class Start,class Shared,class Resident,class JoinGateUp,class GateUp,class JoinDown,class Down,class Reduce>
auto execute_routed_moe(Start&& start,Shared&& shared,Resident&& resident,JoinGateUp&& join_gate_up,
        GateUp&& gate_up,JoinDown&& join_down,Down&& down,Reduce&& reduce) {
    start();shared();resident();join_gate_up();gate_up();join_down();down();return reduce();
}
} // namespace mfq::cuda
