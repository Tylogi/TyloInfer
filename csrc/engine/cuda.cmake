if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_library(mfq-cuda-storage STATIC
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/prepare/expert_store.cpp
)
add_library(mfq::cuda-storage ALIAS mfq-cuda-storage)
target_include_directories(mfq-cuda-storage PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mfq-cuda-storage PUBLIC Threads::Threads)
target_compile_features(mfq-cuda-storage PUBLIC cxx_std_20)
set_target_properties(mfq-cuda-storage PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
    target_compile_options(mfq-cuda-storage PRIVATE /utf-8 /EHsc)
endif()

else()
    add_library(mfq-cuda-storage INTERFACE)
    add_library(mfq::cuda-storage ALIAS mfq-cuda-storage)
endif()

# Engine assembly uses model and weight dependencies. Kernel/quant
# ops have no Engine, model-family, scheduler or transport link.
function(mfq_configure_cuda_engine_target target)
    mfq_configure_cuda_backend_target(${target})
    target_link_libraries(${target} PRIVATE
        mfq-model-source
        mfq-quant-readers
        mfq-engine
        mfq-host-memory
        mfq-prefix-blocks
        mfq-qsa-kv-store
        mfq-cpu-execution
    )
endfunction()

set(TYLOINFER_CUDA_LOADING_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/image.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/routed.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/encoding/rows.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/rows/source.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/rows/cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/rows/prefetch.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/rows/tq.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/rows/gguf.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/rows/gguf_read.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/rows/stage.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/rows/rows.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/loader.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/cache/routed.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/cache/ggml.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/cache/workspace.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/group.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/format.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/prepare/quant.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/prepare/channel_fp8.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/prepare/gguf_weight.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/experts.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/batch.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/load.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_CUDA_LOADING_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/residency/expert_cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/prepare/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/prepare/prepare.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/prepare/quant_range_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/host/cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/residency/residency.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/rows/nint_pipeline.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/prepare/nint.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/prepare/vq.cpp
    )
else()
    list(APPEND TYLOINFER_CUDA_LOADING_SOURCES
        ${PROJECT_SOURCE_DIR}/engine/weights/moe/disabled.cpp)
endif()
add_library(mfq-cuda-loading OBJECT ${TYLOINFER_CUDA_LOADING_SOURCES})
mfq_configure_cuda_engine_target(mfq-cuda-loading)
target_compile_definitions(mfq-cuda-loading PRIVATE NOMINMAX TYLOINFER_HAS_CUDA=1 TYLOINFER_HAS_METAL=0)
set_target_properties(mfq-cuda-loading PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(mfq-cuda-models OBJECT ${TYLOINFER_MODELS_CUDA_SOURCES})
mfq_configure_cuda_backend_target(mfq-cuda-models)
target_link_libraries(mfq-cuda-models PRIVATE mfq-engine)
target_compile_definitions(mfq-cuda-models PRIVATE NOMINMAX)
set_target_properties(mfq-cuda-models PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Library loading calls the model factory and prepares execution resources.
# The unified model factory names concrete families, so it belongs to the CUDA
# composition target rather than the generic Engine.
set(TYLOINFER_CUDA_ENGINE_SOURCES
    ${TYLOINFER_MODEL_ASSEMBLY_SOURCES}
    ${PROJECT_SOURCE_DIR}/engine/models/minicpmo45/duplex.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/minicpmo45/engine.cpp
    ${PROJECT_SOURCE_DIR}/engine/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/layer_stream.cpp
    ${TYLOINFER_CUDA_MODEL_SOURCES}
)
add_library(mfq-cuda-engine STATIC
    $<TARGET_OBJECTS:mfq-cuda-ops>
    $<TARGET_OBJECTS:mfq-cuda-execution>
    $<TARGET_OBJECTS:mfq-cuda-loading>
    $<TARGET_OBJECTS:mfq-cuda-models>
    ${TYLOINFER_CUDA_ENGINE_SOURCES}
)
add_library(mfq::cuda-engine ALIAS mfq-cuda-engine)
mfq_configure_cuda_engine_target(mfq-cuda-engine)
target_link_libraries(mfq-cuda-engine PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    mfq-cuda-core
    mfq-cuda-native-kernels
    mfq-cuda-storage
)
target_compile_definitions(mfq-cuda-engine PRIVATE NOMINMAX TYLOINFER_HAS_CUDA=1 TYLOINFER_HAS_METAL=0)
if(MSVC)
    foreach(target mfq-cuda-ops mfq-cuda-execution mfq-cuda-loading mfq-cuda-models mfq-cuda-engine)
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
            "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"
            "$<$<COMPILE_LANGUAGE:CXX>:/bigobj>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
        )
    endforeach()
endif()
set_target_properties(mfq-cuda-engine PROPERTIES
    CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
)

# The production entry point is part of the CUDA engine build.
include(${PROJECT_SOURCE_DIR}/apps/cuda.cmake)
