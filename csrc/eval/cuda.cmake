add_library(mfq-cuda-eval STATIC
    ${PROJECT_SOURCE_DIR}/eval/minicpmo45.cpp
    ${PROJECT_SOURCE_DIR}/eval/tensors.cpp
)
mfq_configure_cuda_engine_target(mfq-cuda-eval)
target_link_libraries(mfq-cuda-eval PUBLIC mfq-cuda-engine)
