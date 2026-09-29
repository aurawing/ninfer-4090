include_guard(GLOBAL)

option(NINFER_BUILD_CPU_VISION "Build the pinned GGML CPU vision backend" OFF)
set(NINFER_GGML_SOURCE_DIR "" CACHE PATH "Complete source tree of pinned llama.cpp (copied before private patching)")
if(NOT DEFINED NINFER_SOURCE_ROOT)
    get_filename_component(NINFER_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

# Function scope contains upstream options and output-directory changes. In particular,
# CUDA, BUILD_SHARED_LIBS and GGML options do not change NInfer's other dependencies.
function(ninfer_add_cpu_vision)
    set(_vision "${NINFER_SOURCE_ROOT}/src/targets/qwen3_6/impl/vision")
    add_library(ninfer_cpu_vision STATIC "${_vision}/cpu_vision_bridge.cpp" "${_vision}/cpu_vision_cache.cpp")
    target_include_directories(ninfer_cpu_vision PUBLIC "${NINFER_SOURCE_ROOT}/src")
    target_compile_features(ninfer_cpu_vision PUBLIC cxx_std_20)
    if(NOT NINFER_BUILD_CPU_VISION)
        target_sources(ninfer_cpu_vision PRIVATE "${_vision}/cpu_vision_encoder_stub.cpp")
        return()
    endif()

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(_revision b81c99b479d4c24e5eeca10de99032ebd343ef8f)
    set(_private_source "${CMAKE_BINARY_DIR}/_deps/ninfer-llama-${_revision}")
    if(NINFER_GGML_SOURCE_DIR)
        if(NOT EXISTS "${NINFER_GGML_SOURCE_DIR}/tools/mtmd/clip.cpp" OR
           NOT EXISTS "${NINFER_GGML_SOURCE_DIR}/vendor/CMakeLists.txt")
            message(FATAL_ERROR "NINFER_GGML_SOURCE_DIR must be the complete pinned llama.cpp source tree")
        endif()
        if(NOT EXISTS "${_private_source}/.ninfer-source-origin")
            file(MAKE_DIRECTORY "${_private_source}")
            file(COPY "${NINFER_GGML_SOURCE_DIR}/" DESTINATION "${_private_source}")
            file(WRITE "${_private_source}/.ninfer-source-origin" "${NINFER_GGML_SOURCE_DIR}")
        endif()
        file(READ "${_private_source}/.ninfer-source-origin" _origin)
        if(NOT _origin STREQUAL NINFER_GGML_SOURCE_DIR)
            message(FATAL_ERROR "CPU vision source override changed; use a fresh build directory")
        endif()
    else()
        include(FetchContent)
        FetchContent_Declare(ninfer_llama_source
            URL "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/${_revision}"
            URL_HASH SHA256=cde31415ed11844016b5d99906d67335632f04cd45e0b2ae8891ea5e5650ee4c
            SOURCE_DIR "${_private_source}"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        # Populate only: the source must be patched before adding upstream targets.
        if(POLICY CMP0169)
            cmake_policy(SET CMP0169 OLD)
        endif()
        FetchContent_GetProperties(ninfer_llama_source)
        if(NOT ninfer_llama_source_POPULATED)
            FetchContent_Populate(ninfer_llama_source)
        endif()
    endif()
    execute_process(COMMAND "${Python3_EXECUTABLE}"
        "${NINFER_SOURCE_ROOT}/tools/dependencies/patch_ggml_cpu_vision.py" "${_private_source}"
        RESULT_VARIABLE _patch_result ERROR_VARIABLE _patch_error)
    if(NOT _patch_result EQUAL 0)
        message(FATAL_ERROR "CPU vision pinned dependency patch failed: ${_patch_error}")
    endif()

    cmake_policy(SET CMP0077 NEW)
    set(BUILD_SHARED_LIBS OFF)
    set(GGML_CPU ON)
    set(GGML_NATIVE OFF)
    set(GGML_AVX ON)
    set(GGML_AVX2 ON)
    set(GGML_FMA ON)
    set(GGML_F16C ON)
    set(GGML_OPENMP ON)
    set(GGML_LLAMAFILE ON)
    foreach(_flag IN ITEMS GGML_CUDA GGML_HIP GGML_MUSA GGML_VULKAN GGML_METAL GGML_WEBGPU
            GGML_SYCL GGML_OPENCL GGML_CANN GGML_ZDNN GGML_VIRTGPU GGML_RPC GGML_BLAS
            GGML_ACCELERATE GGML_BACKEND_DL GGML_CPU_ALL_VARIANTS
            GGML_AVX_VNNI GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16
            GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
        set(${_flag} OFF)
    endforeach()
    foreach(_flag IN ITEMS LLAMA_BUILD_COMMON LLAMA_BUILD_TOOLS LLAMA_BUILD_SERVER LLAMA_BUILD_APP
            LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_UI LLAMA_SUBPROCESS MTMD_VIDEO
            LLAMA_TOOLS_INSTALL LLAMA_CURL GGML_BUILD_TESTS GGML_BUILD_EXAMPLES)
        set(${_flag} OFF)
    endforeach()
    set(LLAMA_BUILD_MTMD ON)
    # The pinned llama text support target uses u8 literals with char interfaces.
    # Keep its upstream C++17 contract scoped to this dependency; NInfer and its
    # wrapper target retain their C++20 requirement.
    set(CMAKE_CXX_STANDARD 17)
    add_subdirectory("${_private_source}" "${CMAKE_BINARY_DIR}/_deps/ninfer-llama-build" EXCLUDE_FROM_ALL)
    target_sources(ninfer_cpu_vision PRIVATE "${_vision}/ggml_cpu_vision_encoder.cpp")
    target_link_libraries(ninfer_cpu_vision PRIVATE mtmd)
    set(NINFER_CPU_VISION_GGML_SOURCE_DIR "${_private_source}" PARENT_SCOPE)
endfunction()

ninfer_add_cpu_vision()
