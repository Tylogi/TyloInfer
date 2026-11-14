# Backend-independent native tests. Keep assert checks in Release too.
# Same include roots as the mfq-core interface every test links against.
include_directories(${PROJECT_SOURCE_DIR})
if(MSVC)
    add_compile_options(/UNDEBUG $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
else()
    add_compile_options(-UNDEBUG)
endif()

add_executable(tyloinfer-benchmark-validation-test benchmark_validation_test.cpp)
target_include_directories(tyloinfer-benchmark-validation-test PRIVATE
    ${TYLOINFER_REPOSITORY_ROOT}/bench
)
target_compile_features(tyloinfer-benchmark-validation-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-benchmark-validation-test COMMAND tyloinfer-benchmark-validation-test)

add_executable(tyloinfer-cuda-tensor-view-test
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/tests/mfq_cuda_tensor_view_test.cpp
)
target_include_directories(tyloinfer-cuda-tensor-view-test PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor
)
target_compile_features(tyloinfer-cuda-tensor-view-test PRIVATE cxx_std_20)
add_test(NAME tyloinfer-cuda-tensor-view-test COMMAND tyloinfer-cuda-tensor-view-test)

add_executable(tyloinfer-native-tensor-test
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/env.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/tensor.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/tests/mfq_native_tensor_cpu_test_stubs.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/tests/mfq_native_tensor_test.cpp
)
target_include_directories(tyloinfer-native-tensor-test PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor
)
target_compile_features(tyloinfer-native-tensor-test PRIVATE cxx_std_20)
add_test(NAME tyloinfer-native-tensor-test COMMAND tyloinfer-native-tensor-test)

add_executable(tyloinfer-paged-kv-allocator-test
    paged_kv_allocator_test.cpp
)
target_link_libraries(tyloinfer-paged-kv-allocator-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-paged-kv-allocator-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-paged-kv-allocator-test COMMAND tyloinfer-paged-kv-allocator-test)

add_executable(tyloinfer-session-snapshot-cache-test
    session_snapshot_cache_test.cpp
)
target_link_libraries(tyloinfer-session-snapshot-cache-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-session-snapshot-cache-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-session-snapshot-cache-test
    COMMAND tyloinfer-session-snapshot-cache-test)

function(tyloinfer_add_core_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE mfq-core)
    target_compile_features(${target} PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME ${target} COMMAND ${target})
endfunction()

tyloinfer_add_core_test(tyloinfer-tq-rows-test tq_rows_test.cpp)
tyloinfer_add_core_test(tyloinfer-qsa-precision-policy-test qsa_precision_policy_test.cpp)
tyloinfer_add_core_test(tyloinfer-qsa-ram-policy-test qsa_ram_policy_test.cpp)
tyloinfer_add_core_test(tyloinfer-qsa-ram-codec-test qsa_ram_codec_test.cpp)
target_link_libraries(tyloinfer-qsa-ram-codec-test PRIVATE mfq-ggml)
tyloinfer_add_core_test(tyloinfer-qsa-uniform-fit-test qsa_uniform_fit_test.cpp)
target_link_libraries(tyloinfer-qsa-uniform-fit-test PRIVATE mfq-ggml)
add_executable(tyloinfer-qsa-adaptive-cache-test qsa_adaptive_cache_test.cpp)
target_link_libraries(tyloinfer-qsa-adaptive-cache-test PRIVATE mfq-core)
target_compile_features(tyloinfer-qsa-adaptive-cache-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-qsa-adaptive-cache-test COMMAND tyloinfer-qsa-adaptive-cache-test
    "${CMAKE_CURRENT_BINARY_DIR}/qsa-adaptive-cache-test")
set_tests_properties(tyloinfer-qsa-adaptive-cache-test PROPERTIES TIMEOUT 30
    ENVIRONMENT "TYLOINFER_DISK_BUDGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/qsa-adaptive-disk-budget-test")
tyloinfer_add_core_test(tyloinfer-row-residual-source-test row_residual_test.cpp)
target_link_libraries(tyloinfer-row-residual-source-test PRIVATE mfq-ggml)
tyloinfer_add_core_test(tyloinfer-format-features-test format_features_test.cpp)

add_executable(tyloinfer-text-emitter-test text_emitter_test.cpp)
target_link_libraries(tyloinfer-text-emitter-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-text-emitter-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-text-emitter-test COMMAND tyloinfer-text-emitter-test)

add_executable(tyloinfer-prefill-budget-test prefill_budget_test.cpp)
target_link_libraries(tyloinfer-prefill-budget-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-prefill-budget-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-prefill-budget-test COMMAND tyloinfer-prefill-budget-test)

if(TARGET mfq-tokenizer)
    add_executable(tyloinfer-tokenizer-identity-test tokenizer_identity_test.cpp)
    target_link_libraries(tyloinfer-tokenizer-identity-test PRIVATE mfq-engine mfq-ggml)
    target_compile_features(tyloinfer-tokenizer-identity-test PRIVATE cxx_std_20)
    add_test(NAME tyloinfer-tokenizer-identity-test COMMAND tyloinfer-tokenizer-identity-test)

    add_executable(tyloinfer-text-processor-test text_processor_test.cpp)
    target_link_libraries(tyloinfer-text-processor-test PRIVATE mfq-engine)
    target_compile_features(tyloinfer-text-processor-test PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(tyloinfer-text-processor-test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME tyloinfer-text-processor-test COMMAND tyloinfer-text-processor-test)
endif()

tyloinfer_add_core_test(tyloinfer-model-graph-test model_graph_test.cpp)
tyloinfer_add_core_test(tyloinfer-model-source-test model_source_test.cpp)
add_executable(tyloinfer-mlx-affine-source-test mlx_affine_model_source_test.cpp)
target_include_directories(tyloinfer-mlx-affine-source-test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(tyloinfer-mlx-affine-source-test PRIVATE mfq-model-source mfq-json)
add_test(NAME tyloinfer-mlx-affine-source-test COMMAND tyloinfer-mlx-affine-source-test)
add_executable(tyloinfer-gguf-model-source-test gguf_model_source_test.cpp)
target_include_directories(tyloinfer-gguf-model-source-test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(tyloinfer-gguf-model-source-test PRIVATE mfq-model-source mfq-json mfq-ggml)
add_test(NAME tyloinfer-gguf-model-source-test COMMAND tyloinfer-gguf-model-source-test)
tyloinfer_add_core_test(tyloinfer-file-range-reader-test file_range_reader_test.cpp)
tyloinfer_add_core_test(tyloinfer-mirrored-file-reader-test mirrored_file_reader_test.cpp)
tyloinfer_add_core_test(tyloinfer-moe-dispatch-plan-test moe_dispatch_plan_test.cpp)
if(TYLOINFER_BUILD_BENCHES)
add_executable(mfq-moe-residency-bench ${PROJECT_SOURCE_DIR}/../bench/moe_residency_bench.cpp)
target_link_libraries(mfq-moe-residency-bench PRIVATE mfq-core)
target_compile_features(mfq-moe-residency-bench PRIVATE cxx_std_17)
endif()
tyloinfer_add_core_test(tyloinfer-nint-rows-test nint_rows_test.cpp)
tyloinfer_add_core_test(tyloinfer-quant-expert-range-test quant_expert_range_test.cpp)
tyloinfer_add_core_test(tyloinfer-host-parallel-test host_parallel_test.cpp)
tyloinfer_add_core_test(tyloinfer-cpu-expert-pool-test cpu_expert_pool_test.cpp)
target_link_libraries(tyloinfer-cpu-expert-pool-test PRIVATE mfq-cpu-execution)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
tyloinfer_add_core_test(tyloinfer-packed-nint-cpu-test packed_nint_cpu_test.cpp)
target_link_libraries(tyloinfer-packed-nint-cpu-test PRIVATE mfq-cpu-ops)
tyloinfer_add_core_test(tyloinfer-nvq-group-cpu-test nvq_group_cpu_test.cpp)
target_link_libraries(tyloinfer-nvq-group-cpu-test PRIVATE mfq-cpu-ops)
endif()
tyloinfer_add_core_test(tyloinfer-grid-vision-test grid_vision_test.cpp)
target_link_libraries(tyloinfer-grid-vision-test PRIVATE mfq-engine)

# RADIO cropped-position geometry: backend-neutral, no CUDA or weights needed.
add_executable(tyloinfer-nemotron-vision-geometry-test nemotron_vision_geometry_test.cpp)
target_link_libraries(tyloinfer-nemotron-vision-geometry-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-nemotron-vision-geometry-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-nemotron-vision-geometry-test
    COMMAND tyloinfer-nemotron-vision-geometry-test)

add_executable(tyloinfer-nemotron-vision-config-test nemotron_vision_config_test.cpp)
target_link_libraries(tyloinfer-nemotron-vision-config-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-nemotron-vision-config-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-nemotron-vision-config-test
    COMMAND tyloinfer-nemotron-vision-config-test)
tyloinfer_add_core_test(tyloinfer-deepseek-v41-config-test deepseek_v41_config_test.cpp)
target_link_libraries(tyloinfer-deepseek-v41-config-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-deepseek-v41-config-test PRIVATE cxx_std_20)
tyloinfer_add_core_test(tyloinfer-model-config-test model_config_test.cpp)
target_link_libraries(tyloinfer-model-config-test PRIVATE mfq-engine)
add_executable(tyloinfer-sampling-test sampling_test.cpp)
target_link_libraries(tyloinfer-sampling-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-sampling-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-sampling-test COMMAND tyloinfer-sampling-test)
add_executable(tyloinfer-moe-cache-policy-test moe_cache_policy_test.cpp)
target_link_libraries(tyloinfer-moe-cache-policy-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-moe-cache-policy-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-moe-cache-policy-test COMMAND tyloinfer-moe-cache-policy-test)
tyloinfer_add_core_test(tyloinfer-tensor-parallel-test tensor_parallel_test.cpp)

add_executable(tyloinfer-cuda-startup-test cuda_startup_test.cpp)
target_include_directories(tyloinfer-cuda-startup-test PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(tyloinfer-cuda-startup-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-cuda-startup-test COMMAND tyloinfer-cuda-startup-test)

add_executable(tyloinfer-cuda-model-plan-test
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/tests/cuda_model_plan_test.cpp
)
target_include_directories(tyloinfer-cuda-model-plan-test PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(tyloinfer-cuda-model-plan-test PRIVATE mfq-core)
target_compile_features(tyloinfer-cuda-model-plan-test PRIVATE cxx_std_20)
add_test(NAME tyloinfer-cuda-model-plan-test COMMAND tyloinfer-cuda-model-plan-test)

add_executable(tyloinfer-moe-host-memory-test
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/tests/moe_host_memory_test.cpp
)
target_include_directories(tyloinfer-moe-host-memory-test PRIVATE
    ${PROJECT_SOURCE_DIR}/backends/cuda
)
target_compile_features(tyloinfer-moe-host-memory-test PRIVATE cxx_std_20)
target_link_libraries(tyloinfer-moe-host-memory-test PRIVATE mfq-host-parallel)
add_test(NAME tyloinfer-moe-host-memory-test COMMAND tyloinfer-moe-host-memory-test)

if(TARGET mfq-tokenizer)
add_executable(tyloinfer-inference-test inference_test.cpp)
target_link_libraries(tyloinfer-inference-test PRIVATE mfq-engine)
add_test(NAME tyloinfer-inference-test COMMAND tyloinfer-inference-test)

add_executable(tyloinfer-request-executor-test request_executor_test.cpp)
target_link_libraries(tyloinfer-request-executor-test PRIVATE mfq-engine)
add_test(NAME tyloinfer-request-executor-test COMMAND tyloinfer-request-executor-test)

add_executable(tyloinfer-shared-execution-test shared_execution_test.cpp)
target_link_libraries(tyloinfer-shared-execution-test PRIVATE mfq-engine mfq-engine)
add_test(NAME tyloinfer-shared-execution-test COMMAND tyloinfer-shared-execution-test)
endif()

add_executable(tyloinfer-mtp-policy-test mtp_policy_test.cpp)
target_link_libraries(tyloinfer-mtp-policy-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-mtp-policy-test PRIVATE cxx_std_20)
add_test(NAME tyloinfer-mtp-policy-test COMMAND tyloinfer-mtp-policy-test)

add_executable(tyloinfer-speculative-commit-test speculative_commit_test.cpp)
target_link_libraries(tyloinfer-speculative-commit-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-speculative-commit-test PRIVATE cxx_std_20)
add_test(NAME tyloinfer-speculative-commit-test COMMAND tyloinfer-speculative-commit-test)

if(TARGET mfq-tokenizer)
add_executable(tyloinfer-speculative-sequence-test speculative_sequence_test.cpp)
target_link_libraries(tyloinfer-speculative-sequence-test PRIVATE mfq-engine)
target_compile_features(tyloinfer-speculative-sequence-test PRIVATE cxx_std_20)
add_test(NAME tyloinfer-speculative-sequence-test COMMAND tyloinfer-speculative-sequence-test)
endif()

add_executable(tyloinfer-moe-cache-profile-test moe_cache_profile_test.cpp)
target_link_libraries(tyloinfer-moe-cache-profile-test PRIVATE mfq-core mfq-json)
target_compile_features(tyloinfer-moe-cache-profile-test PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(tyloinfer-moe-cache-profile-test PRIVATE /utf-8 /EHsc)
endif()
add_test(NAME tyloinfer-moe-cache-profile-test COMMAND tyloinfer-moe-cache-profile-test)

if(TARGET mfq-paged-prefix-cache)
    add_executable(tyloinfer-paged-prefix-cache-test
        paged_prefix_cache_test.cpp
    )
    target_link_libraries(tyloinfer-paged-prefix-cache-test PRIVATE
        mfq-engine
        mfq-paged-prefix-cache
    )
    target_compile_features(tyloinfer-paged-prefix-cache-test PRIVATE cxx_std_17)
    add_test(NAME tyloinfer-paged-prefix-cache-test COMMAND tyloinfer-paged-prefix-cache-test)
endif()

if(TARGET mfq-runtime-communication)
    tyloinfer_add_core_test(tyloinfer-runtime-transport-test runtime_transport_test.cpp)
    target_link_libraries(tyloinfer-runtime-transport-test PRIVATE mfq-runtime-communication mfq-json)
    target_include_directories(tyloinfer-runtime-transport-test PRIVATE
        ${PROJECT_SOURCE_DIR}/transport/src
    )

    add_executable(tyloinfer-runtime-profile-test runtime_profile_test.cpp)
    target_link_libraries(tyloinfer-runtime-profile-test PRIVATE mfq-runtime-communication)
    target_compile_features(tyloinfer-runtime-profile-test PRIVATE cxx_std_17)
    add_test(NAME tyloinfer-runtime-profile-test COMMAND tyloinfer-runtime-profile-test)
endif()

add_executable(tyloinfer-deepseek-v41-vision-layout-test deepseek_v41_vision_layout_test.cpp)
target_link_libraries(tyloinfer-deepseek-v41-vision-layout-test PRIVATE mfq-engine)
add_test(NAME tyloinfer-deepseek-v41-vision-layout-test COMMAND tyloinfer-deepseek-v41-vision-layout-test)

# The split policy is pure host arithmetic and is testable without CUDA or a GPU.
add_executable(tyloinfer-qsa-ram-store-test qsa_ram_store_test.cpp)
target_link_libraries(tyloinfer-qsa-ram-store-test PRIVATE mfq-qsa-kv-store)
if(TARGET mfq-paged-prefix-cache)
    target_link_libraries(tyloinfer-qsa-ram-store-test PRIVATE mfq-paged-prefix-cache)
    target_compile_definitions(tyloinfer-qsa-ram-store-test PRIVATE TYLOINFER_TEST_PREFIX_BORROW=1)
endif()
target_compile_features(tyloinfer-qsa-ram-store-test PRIVATE cxx_std_17)
add_test(NAME tyloinfer-qsa-ram-store-test COMMAND tyloinfer-qsa-ram-store-test ${CMAKE_CURRENT_BINARY_DIR}/qsa-ram-store-test-data)

add_executable(tyloinfer-gguf-cpu-balance-test gguf_cpu_balance_test.cpp)
target_include_directories(tyloinfer-gguf-cpu-balance-test PRIVATE ${PROJECT_SOURCE_DIR})
find_package(Threads REQUIRED)
target_link_libraries(tyloinfer-gguf-cpu-balance-test PRIVATE Threads::Threads)
target_compile_features(tyloinfer-gguf-cpu-balance-test PRIVATE cxx_std_20)
add_test(NAME tyloinfer-gguf-cpu-balance-test COMMAND tyloinfer-gguf-cpu-balance-test)
