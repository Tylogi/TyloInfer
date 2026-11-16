cmake_minimum_required(VERSION 3.26)
project(mfq_cpp_runtime LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TYLOINFER_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

option(TYLOINFER_BUILD_SERVER "Build the native public server" OFF)

option(TYLOINFER_ENABLE_MFQ_COMPAT "Include legacy MFQ inference formats" OFF)
option(
    TYLOINFER_BUILD_METAL_RUNTIME
    "Build the native MLX/Metal runtime on Apple Silicon"
    ${APPLE}
)

include(CheckLanguage)
check_language(CUDA)
if(NOT APPLE AND CMAKE_CUDA_COMPILER AND NOT TYLOINFER_BUILD_HIP_RUNTIME)
    set(_TYLOINFER_BUILD_CUDA_ENGINE_DEFAULT ON)
else()
    set(_TYLOINFER_BUILD_CUDA_ENGINE_DEFAULT OFF)
endif()
option(
    TYLOINFER_BUILD_CUDA_ENGINE
    "Build the native CUDA Engine (requires the CUDA toolkit)"
    ${_TYLOINFER_BUILD_CUDA_ENGINE_DEFAULT}
)
unset(_TYLOINFER_BUILD_CUDA_ENGINE_DEFAULT)

check_language(HIP)
if(NOT APPLE AND CMAKE_HIP_COMPILER AND NOT TYLOINFER_BUILD_CUDA_ENGINE)
    set(_TYLOINFER_BUILD_HIP_RUNTIME_DEFAULT ON)
else()
    set(_TYLOINFER_BUILD_HIP_RUNTIME_DEFAULT OFF)
endif()
option(
    TYLOINFER_BUILD_HIP_RUNTIME
    "Build the native HIP runtime on AMD ROCm hosts"
    ${_TYLOINFER_BUILD_HIP_RUNTIME_DEFAULT}
)
unset(_TYLOINFER_BUILD_HIP_RUNTIME_DEFAULT)

if(TYLOINFER_BUILD_CUDA_ENGINE AND TYLOINFER_BUILD_HIP_RUNTIME)
    message(FATAL_ERROR
        "Select either TYLOINFER_BUILD_CUDA_ENGINE or TYLOINFER_BUILD_HIP_RUNTIME; use separate build directories for both backends")
endif()

if(TYLOINFER_BUILD_CUDA_ENGINE AND TYLOINFER_BUILD_METAL_RUNTIME)
    message(FATAL_ERROR
        "Select either TYLOINFER_BUILD_CUDA_ENGINE or TYLOINFER_BUILD_METAL_RUNTIME; use separate build directories for both backends")
endif()

option(
    TYLOINFER_BUILD_TRANSPORT
    "Build private HTTP/stdio transport for Engine applications"
    ON
)

if(TYLOINFER_BUILD_METAL_RUNTIME AND NOT APPLE)
    message(FATAL_ERROR "TYLOINFER_BUILD_METAL_RUNTIME requires macOS")
endif()

# engine/ links CUDA::cudart and CUDA::cublas, so the toolkit targets must exist
# before the shared Engine directory is processed.
if(TYLOINFER_BUILD_CUDA_ENGINE)
    find_package(CUDAToolkit REQUIRED)
endif()

# Common code is split by responsibility. JSON is defined first because the
# canonical model graph and several public policy headers consume it.
add_subdirectory(components/json)
add_library(mfq-host-memory STATIC components/host_memory.cpp)
target_include_directories(mfq-host-memory PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(mfq-host-memory PUBLIC cxx_std_17)
set_target_properties(mfq-host-memory PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(WIN32)
    target_link_libraries(mfq-host-memory PRIVATE psapi)
endif()
if(MSVC)
    target_compile_options(mfq-host-memory PRIVATE /EHsc)
endif()
add_subdirectory(engine/weights/encoding)
add_subdirectory(engine/models/source)
add_subdirectory(engine/backends/cpu)
add_subdirectory(engine/state)
add_subdirectory(components/io)
add_subdirectory(engine/weights/source)
# Engine request/text headers include the tokenizer chat API even for
# communication-free CUDA/CPU consumers. Keep that real dependency available
# independently of the transport modules.
add_subdirectory(components/ggml)
add_subdirectory(components/tokenizer)

if(TYLOINFER_BUILD_TRANSPORT OR TYLOINFER_BUILD_SERVER)
    add_subdirectory(components/http)
endif()

add_subdirectory(engine)
add_subdirectory(engine/models)

if(TYLOINFER_BUILD_TRANSPORT OR TYLOINFER_BUILD_SERVER)
    add_subdirectory(components/protocol)
endif()

if(TYLOINFER_BUILD_TRANSPORT)
    add_subdirectory(transport)
endif()

if(TYLOINFER_BUILD_METAL_RUNTIME)
    enable_language(OBJCXX)
    # Keep the established build output path and executable locations.
    add_subdirectory(engine/backends/metal metal)
endif()

if(TYLOINFER_BUILD_CUDA_ENGINE)
    add_subdirectory(engine/backends/cuda cuda)
endif()

if(TYLOINFER_BUILD_HIP_RUNTIME)
    add_subdirectory(backends/hip hip)
endif()

# Umbrella for the backend-independent core modules that test targets link.
add_library(mfq-core INTERFACE)
add_library(mfq::core ALIAS mfq-core)
target_include_directories(mfq-core INTERFACE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/engine
)
target_link_libraries(mfq-core INTERFACE
    mfq-host-memory
    mfq-host-parallel
    mfq-model-graph
    mfq-model-source
    mfq-json
    mfq-qsa-kv-store
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(TYLOINFER_BUILD_SERVER)
    add_subdirectory(server)
endif()
