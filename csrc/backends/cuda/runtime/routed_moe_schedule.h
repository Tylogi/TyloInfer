#pragma once
#include "runtime/decode_window.h"
#include "routed_moe_order.h"
#include <utility>

namespace mfq::cuda {
enum class RoutedImageUse {Eager,Decode,Admission,Restore};

// GGUF's graph lifecycle serves mapped routes within one submission and uses
// host boundaries only for unmapped transfers. Warmup consumes real weights;
// every format supplies its own immutable plans and release callbacks.
template<class Wait,class Warmup,class Serve>
void prepare_routed_graph(DecodeWindow* window,cudaStreamCaptureStatus capture,bool mapped,bool direct,
        DecodeWindow::Task task,Wait wait,Warmup warmup,Serve serve) {
    if(capture==cudaStreamCaptureStatusNone){wait();warmup();}
    if(!window)return;
    task.serve=[mapped,wait,serve]{if(mapped){wait();serve();}};
    window->enroll(std::move(task));
    if(!direct)window->host_barrier([mapped,serve]{if(!mapped)serve();});
}
} // namespace mfq::cuda
