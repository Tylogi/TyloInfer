# Keep storage and load accounting separate from DenseTensor numerical operations.
add_library(mfq-tensor STATIC tensor.cpp views.cpp dense_tensor.cpp weights/memory.cpp)
add_library(mfq::tensor ALIAS mfq-tensor)
target_include_directories(mfq-tensor PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(mfq-tensor PUBLIC cxx_std_20)
set_target_properties(mfq-tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(mfq-engine STATIC
    config.cpp
    storage_ops.cpp
    engine.cpp
    executor.cpp
    generation/decode.cpp
    generation/generation.cpp
    generation/mtp/metrics.cpp
    generation/mtp/generation.cpp
    generation/mtp/sampling.cpp
    models/prepare.cpp
    models/state.cpp
    models/deepseek_v4/input.cpp
    models/minicpmo45/input.cpp
    models/deepseek_v41/axial_patch_vision.cpp
    models/common/attention.cpp
    models/common/ffn.cpp
    models/common/linear.cpp
    models/common/moe.cpp
    models/common/vision/grid_vision.cpp
    models/common/vision/grid_encoder.cpp
    models/deepseek_v4/config.cpp
    models/deepseek_v41/config.cpp
    models/gemma4/config.cpp
    models/glm5_next/config.cpp
    models/glm_dsa/config.cpp
    models/minicpmo45/config.cpp
    models/nemotron_h/config.cpp
    models/nemotron_h/vision.cpp
    models/qwen35/config.cpp
    models/qwen35/batch.cpp
    models/qwen35/model.cpp
    models/qwen35/forward.cpp
    models/qwen35/state.cpp
    models/qwen35/mtp_state.cpp
    models/qwen35/mtp.cpp
    models/qwen35/predictor.cpp
    models/qwen4_exp/config.cpp
    requests/constraint.cpp
    requests/emitter.cpp
    requests/output.cpp
    requests/policy.cpp
    requests/processor.cpp
    requests/tokenizer.cpp
    scheduler/scheduler.cpp
    scheduler/request.cpp
    state/session/paged.cpp
    state/session/prefix_cache.cpp
    state/qsa_sequence.cpp
    state/recurrent.cpp
    state/recurrent_layout.cpp
    state/kv_view.cpp
    state/kv_layout.cpp
)
if(TYLOINFER_BUILD_CUDA_ENGINE)
    target_compile_definitions(mfq-engine PUBLIC TYLOINFER_NATIVE_CUDA=1)
    target_include_directories(mfq-engine PRIVATE
        ${TYLOINFER_REPOSITORY_ROOT}
        ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native
        ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor
        ${PROJECT_SOURCE_DIR}/engine/backends/cuda/kernels)
    target_link_libraries(mfq-engine PRIVATE
        CUDA::cudart CUDA::cublas mfq-ggml-internal-headers)
endif()
if(TYLOINFER_BUILD_METAL_RUNTIME)
    target_compile_definitions(mfq-engine PUBLIC TYLOINFER_NATIVE_MLX_RUNTIME=1)
    target_link_libraries(mfq-engine PRIVATE mfq-metal-ops)
endif()
add_library(mfq::engine ALIAS mfq-engine)
target_include_directories(mfq-engine
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/components/tokenizer/include
        ${PROJECT_SOURCE_DIR}/components/tokenizer/chat
    PRIVATE
        ${PROJECT_SOURCE_DIR}/components/tokenizer
        ${PROJECT_SOURCE_DIR}/components/tokenizer/src
)
target_compile_features(mfq-engine PUBLIC cxx_std_20)
target_link_libraries(mfq-engine PUBLIC
    mfq-tensor
    mfq-model-source
    mfq-model-graph
    mfq-formats
    mfq-json
    mfq-prefix-blocks
    mfq-qsa-kv-store
    mfq-cpu-ops
)
set_target_properties(mfq-engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(mfq-engine PUBLIC mfq-tokenizer)
