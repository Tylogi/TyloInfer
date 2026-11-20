if(TARGET mfq-transport)
    add_executable(tyloi-engine
        ${PROJECT_SOURCE_DIR}/apps/runtime_main.cpp
        ${PROJECT_SOURCE_DIR}/commands/runtime.cpp
        ${PROJECT_SOURCE_DIR}/commands/cli.cpp
    )
    target_link_libraries(tyloi-engine PRIVATE
        mfq-cuda-engine mfq-transport)
    if(CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED)
        target_link_libraries(tyloi-engine PRIVATE
            "$<LINK_GROUP:RESCAN,mfq-cuda-engine,mfq-engine,mfq-cuda-core,mfq-tensor>")
    endif()
    target_compile_definitions(tyloi-engine PRIVATE TYLOINFER_CLI_CUDA=1 TYLOINFER_CLI_METAL=0)
endif()

if(TARGET tyloi-engine)
    mfq_configure_cuda_engine_target(tyloi-engine)
    target_compile_definitions(tyloi-engine PRIVATE NOMINMAX)
    if(MSVC)
        target_compile_options(tyloi-engine PRIVATE /utf-8 /EHsc /bigobj)
    endif()
    set_target_properties(tyloi-engine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    )
endif()

if(NOT WIN32 AND TARGET tyloi-engine)
    set(TYLOINFER_SYSTEM_RUNTIME_DIR
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
        "directory containing the host libstdc++ runtime")
    set_property(TARGET tyloi-engine PROPERTY
        BUILD_RPATH "${TYLOINFER_SYSTEM_RUNTIME_DIR}")
endif()
