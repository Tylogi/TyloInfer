add_library(mfq-formats INTERFACE)
target_include_directories(mfq-formats INTERFACE
    ${PROJECT_SOURCE_DIR})

add_library(mfq-quant-readers STATIC
    features.cpp
    gguf/decode.cpp
    gguf/decode.h
    bytes.h
    legacy.h
    dense/blob.cpp
    dense/blob.h
    sq/fp8_blob.cpp
    sq/fp8_blob.h
    quant/nint_rows.cpp
    quant/nint_blob.cpp
    quant/nint_blob.h
    quant/nvq_rows.cpp
    quant/nvq_dense.h
    quant/vq_blob.cpp
    quant/vq_blob.h
    quant/mfe_quant_expert_store.cpp
    quant/mfe_index.cpp
    quant/mfe_index.h
    quant/mfe_blob.cpp
    quant/mfe_blob.h
    mx/affine.cpp
    mx/affine.h
    mx/blob.cpp
    mx/blob.h
    tq/table.cpp
    tq/table.h
    # QSA codecs live in tensor/encoding; mutable KV policy is built by state.
    ${PROJECT_SOURCE_DIR}/engine/weights/encoding/readers/mirrored.cpp
    ${PROJECT_SOURCE_DIR}/engine/tensor/encoding/qsa/tq_cpu.cpp
)
target_include_directories(mfq-quant-readers PRIVATE
    ${PROJECT_SOURCE_DIR})
target_link_libraries(mfq-quant-readers PUBLIC mfq-formats mfq-io
    PRIVATE mfq-host-parallel mfq-json mfq-ggml-internal-headers mfq-ggml)
target_compile_features(mfq-quant-readers PUBLIC cxx_std_17 PRIVATE cxx_std_20)
target_compile_definitions(mfq-quant-readers PRIVATE
    TYLOINFER_ENABLE_MFQ_COMPAT=$<BOOL:${TYLOINFER_ENABLE_MFQ_COMPAT}>)
set_target_properties(mfq-quant-readers PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-quant-readers PRIVATE /EHsc)
endif()
