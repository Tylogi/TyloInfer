add_library(mfq-cuda-diagnostics STATIC
    ${PROJECT_SOURCE_DIR}/diagnostics/checks/attention.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/checks/backend.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/checks/linear.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/checks/operators.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/runtime/experts.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/runtime/ffn.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/models/gemma_moe.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/profiling/trace.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/engines.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/data/source.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/data/text.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/models/vision.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/runtime/mtp_bench.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/runtime/mtp.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/runtime/prefill.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/models/qwen/mtp.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/runtime/batch.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/data/tokens.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/data/compare.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/runtime/policy.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/models/qwen/decode.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/models/qwen/variants.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/runtime/session.cpp
)
mfq_configure_cuda_engine_target(mfq-cuda-diagnostics)
target_compile_definitions(mfq-cuda-diagnostics PRIVATE NOMINMAX)
target_link_libraries(mfq-cuda-diagnostics PUBLIC mfq-cuda-engine)

if(MSVC)
    target_compile_options(mfq-cuda-diagnostics PRIVATE /utf-8 /EHsc /bigobj)
endif()
