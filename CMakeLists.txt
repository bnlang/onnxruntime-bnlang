cmake_minimum_required(VERSION 3.20)
project(onnxruntime-bnlang LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------
# Platform triple — used to find the ORT prebuilt under deps/<triple>/
# Presets in CMakePresets.json set BNL_TARGET_TRIPLE explicitly; if no
# preset is used, we fall back to host detection.
# ----------------------------------------------------------------------
if(NOT BNL_TARGET_TRIPLE)
    if(WIN32)
        set(BNL_TARGET_TRIPLE "windows-x64")
    elseif(APPLE)
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            set(BNL_TARGET_TRIPLE "darwin-arm64")
        else()
            set(BNL_TARGET_TRIPLE "darwin-x64")
        endif()
    else()
        set(BNL_TARGET_TRIPLE "linux-x64")
    endif()
endif()
message(STATUS "Target triple: ${BNL_TARGET_TRIPLE}")

# ----------------------------------------------------------------------
# ORT prebuilt — populated by script/install.bnl into deps/<triple>/
#   deps/<triple>/include/        ORT headers
#   deps/<triple>/lib/            ORT import + shared libs
# ----------------------------------------------------------------------
set(ORT_DIR "${CMAKE_SOURCE_DIR}/deps/${BNL_TARGET_TRIPLE}")
if(NOT EXISTS "${ORT_DIR}/include")
    message(FATAL_ERROR
        "ORT prebuilt not found at ${ORT_DIR}. Run 'bnl script/install.bnl' "
        "(or the platform build script) before configuring.")
endif()

# ----------------------------------------------------------------------
# Native plugin
# ----------------------------------------------------------------------
add_library(onnxruntime-bnlang SHARED
    src/main.cpp
    src/session.cpp
    src/tensor.cpp
    src/session_options.cpp
)

target_include_directories(onnxruntime-bnlang PRIVATE
    src
    "${ORT_DIR}/include"
)

# Link to ORT. Filename differs per platform — keep cmake aware.
if(WIN32)
    target_link_libraries(onnxruntime-bnlang PRIVATE "${ORT_DIR}/lib/onnxruntime.lib")
elseif(APPLE)
    target_link_libraries(onnxruntime-bnlang PRIVATE "${ORT_DIR}/lib/libonnxruntime.dylib")
    set_target_properties(onnxruntime-bnlang PROPERTIES INSTALL_RPATH "@loader_path")
else()
    target_link_libraries(onnxruntime-bnlang PRIVATE "${ORT_DIR}/lib/libonnxruntime.so")
    set_target_properties(onnxruntime-bnlang PROPERTIES INSTALL_RPATH "$ORIGIN/")
endif()

# Land artifact directly under build/<triple>/ with no `lib` prefix on Unix,
# matching the path in bnl.json's `targets` map.
set_target_properties(onnxruntime-bnlang PROPERTIES
    PREFIX                            ""
    RUNTIME_OUTPUT_DIRECTORY          "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY          "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE  "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE  "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG    "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG    "${CMAKE_BINARY_DIR}"
)

# Copy the ORT shared lib next to the plugin so the OS loader finds it.
# onnxruntime_providers_shared.dll ships alongside the DML-enabled build
# and is loaded on demand for non-default execution providers.
if(WIN32)
    add_custom_command(TARGET onnxruntime-bnlang POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_DIR}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:onnxruntime-bnlang>")
    if(EXISTS "${ORT_DIR}/lib/onnxruntime_providers_shared.dll")
        add_custom_command(TARGET onnxruntime-bnlang POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ORT_DIR}/lib/onnxruntime_providers_shared.dll"
                    "$<TARGET_FILE_DIR:onnxruntime-bnlang>")
    endif()
    if(EXISTS "${ORT_DIR}/lib/DirectML.dll")
        add_custom_command(TARGET onnxruntime-bnlang POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ORT_DIR}/lib/DirectML.dll"
                    "$<TARGET_FILE_DIR:onnxruntime-bnlang>")
    endif()
elseif(APPLE)
    add_custom_command(TARGET onnxruntime-bnlang POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_DIR}/lib/libonnxruntime.dylib" "$<TARGET_FILE_DIR:onnxruntime-bnlang>")
else()
    add_custom_command(TARGET onnxruntime-bnlang POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_DIR}/lib/libonnxruntime.so" "$<TARGET_FILE_DIR:onnxruntime-bnlang>")
endif()

# Execution provider flags — stubbed for future use. CPU is always on.
option(USE_DML    "Build with DirectML support" OFF)
option(USE_CUDA   "Build with CUDA support"     OFF)
option(USE_COREML "Build with CoreML support"   OFF)

if(USE_DML)
    target_compile_definitions(onnxruntime-bnlang PRIVATE USE_DML=1)
endif()
if(USE_CUDA)
    target_compile_definitions(onnxruntime-bnlang PRIVATE USE_CUDA=1)
endif()
if(USE_COREML)
    target_compile_definitions(onnxruntime-bnlang PRIVATE USE_COREML=1)
endif()
