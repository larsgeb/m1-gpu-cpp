cmake_minimum_required(VERSION 3.21)
project(m1-gpu-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Platform check ---
if(NOT APPLE)
    message(FATAL_ERROR "This project requires macOS with Apple Silicon (Metal GPU).")
endif()

# --- Metal shader compilation ---
# Find the metal and metallib compilers. On recent macOS (15+) the Metal toolchain
# lives inside a cryptex mount, so we use xcrun to locate them.
execute_process(
    COMMAND xcrun --find metal
    OUTPUT_VARIABLE METAL_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE METAL_FIND_RESULT
)
execute_process(
    COMMAND xcrun --find metallib
    OUTPUT_VARIABLE METALLIB_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE METALLIB_FIND_RESULT
)

if(METAL_FIND_RESULT OR METALLIB_FIND_RESULT OR NOT METAL_COMPILER OR NOT METALLIB_COMPILER)
    message(FATAL_ERROR
        "Could not find Metal shader compilers (metal / metallib).\n"
        "Install Xcode and run: xcodebuild -runFirstLaunch && xcodebuild -downloadComponent MetalToolchain"
    )
endif()
message(STATUS "Metal compiler: ${METAL_COMPILER}")
message(STATUS "Metallib compiler: ${METALLIB_COMPILER}")

# Get the macOS SDK path for Metal shader compilation
execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Function to compile .metal -> .metallib
# TARGET_NAME must be globally unique; OUTPUT_NAME is the filename (e.g. "ops" or "default")
function(compile_metal_shader METAL_SOURCE OUTPUT_DIR OUTPUT_NAME TARGET_NAME)
    get_filename_component(METAL_SOURCE_ABS "${METAL_SOURCE}" ABSOLUTE)
    set(AIR_FILE "${OUTPUT_DIR}/${OUTPUT_NAME}.air")
    set(METALLIB_FILE "${OUTPUT_DIR}/${OUTPUT_NAME}.metallib")

    add_custom_command(
        OUTPUT "${AIR_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E env SDKROOT=${MACOS_SDK_PATH}
                ${METAL_COMPILER} -c "${METAL_SOURCE_ABS}" -o "${AIR_FILE}"
        DEPENDS "${METAL_SOURCE_ABS}"
        COMMENT "Compiling ${METAL_SOURCE} -> AIR"
    )
    add_custom_command(
        OUTPUT "${METALLIB_FILE}"
        COMMAND ${METALLIB_COMPILER} "${AIR_FILE}" -o "${METALLIB_FILE}"
        DEPENDS "${AIR_FILE}"
        COMMENT "Linking ${OUTPUT_NAME}.metallib"
    )
    add_custom_target(${TARGET_NAME} ALL DEPENDS "${METALLIB_FILE}")
endfunction()

# --- Frameworks ---
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)

# --- OpenMP ---
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    # Try Homebrew libomp as fallback
    find_library(OMP_LIB omp HINTS /opt/homebrew/opt/libomp/lib)
    if(OMP_LIB)
        message(STATUS "Found Homebrew libomp: ${OMP_LIB}")
        add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_COMPILE_OPTIONS "-Xclang;-fopenmp"
            INTERFACE_LINK_LIBRARIES "${OMP_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "/opt/homebrew/opt/libomp/include"
        )
    else()
        message(WARNING "OpenMP not found. CPU parallel benchmarks will be disabled.")
    endif()
endif()

# --- metal-cpp include path ---
set(METAL_CPP_DIR "${CMAKE_SOURCE_DIR}/metal-cpp")

# Common compile options for Metal C++ code
function(configure_metal_target TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE ${METAL_CPP_DIR})
    target_compile_options(${TARGET_NAME} PRIVATE -fno-objc-arc)
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
    )
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# =============================================================================
# Example 01 - MetalAdder
# =============================================================================
compile_metal_shader(
    "${CMAKE_SOURCE_DIR}/01-MetalAdder/add.metal"
    "${CMAKE_BINARY_DIR}/01-MetalAdder"
    "default"
    "metallib_01"
)

add_executable(example_01
    01-MetalAdder/main.cpp
    01-MetalAdder/MetalAdder.cpp
)
configure_metal_target(example_01)
add_dependencies(example_01 metallib_01)

# Example 01 uses newDefaultLibrary() which looks for default.metallib in the CWD
set_target_properties(example_01 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/01-MetalAdder"
)

# =============================================================================
# Example 02 - General Array Operations (1D kernels)
# =============================================================================
compile_metal_shader(
    "${CMAKE_SOURCE_DIR}/02-GeneralArrayOperations/ops.metal"
    "${CMAKE_BINARY_DIR}/02-GeneralArrayOperations"
    "ops"
    "metallib_02"
)

add_executable(example_02
    02-GeneralArrayOperations/main.cpp
    02-GeneralArrayOperations/MetalOperations.cpp
    02-GeneralArrayOperations/CPUOperations.cpp
)
configure_metal_target(example_02)
add_dependencies(example_02 metallib_02)
set_target_properties(example_02 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/02-GeneralArrayOperations"
)

# =============================================================================
# Example 03 - 2D Kernels
# =============================================================================
compile_metal_shader(
    "${CMAKE_SOURCE_DIR}/03-2DKernels/ops.metal"
    "${CMAKE_BINARY_DIR}/03-2DKernels"
    "ops"
    "metallib_03"
)

add_executable(example_03
    03-2DKernels/main.cpp
    03-2DKernels/MetalOperations.cpp
    03-2DKernels/CPUOperations.cpp
)
configure_metal_target(example_03)
add_dependencies(example_03 metallib_03)
set_target_properties(example_03 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/03-2DKernels"
)

# =============================================================================
# Example 04 - Compute-heavy kernels (Mandelbrot, N-body)
# =============================================================================
compile_metal_shader(
    "${CMAKE_SOURCE_DIR}/04-Compute/ops.metal"
    "${CMAKE_BINARY_DIR}/04-Compute"
    "ops"
    "metallib_04"
)

# =============================================================================
# Example 05 - Wave Propagation (2D elastic, Virieux staggered grid)
# =============================================================================
compile_metal_shader(
    "${CMAKE_SOURCE_DIR}/05-WavePropagation/ops.metal"
    "${CMAKE_BINARY_DIR}/05-WavePropagation"
    "ops"
    "metallib_05"
)

# =============================================================================
# Tests
# =============================================================================
enable_testing()

add_executable(test_01_adder tests/test_01_adder.cpp 01-MetalAdder/MetalAdder.cpp)
configure_metal_target(test_01_adder)
target_include_directories(test_01_adder PRIVATE "${CMAKE_SOURCE_DIR}/01-MetalAdder")
add_dependencies(test_01_adder metallib_01)
set_target_properties(test_01_adder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/01-MetalAdder"
)
add_test(NAME test_01_adder COMMAND test_01_adder
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/01-MetalAdder"
)

add_executable(test_02_1d_ops
    tests/test_02_1d_ops.cpp
    02-GeneralArrayOperations/MetalOperations.cpp
    02-GeneralArrayOperations/CPUOperations.cpp
)
configure_metal_target(test_02_1d_ops)
target_include_directories(test_02_1d_ops PRIVATE "${CMAKE_SOURCE_DIR}/02-GeneralArrayOperations")
add_dependencies(test_02_1d_ops metallib_02)
set_target_properties(test_02_1d_ops PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/02-GeneralArrayOperations"
)
add_test(NAME test_02_1d_ops COMMAND test_02_1d_ops
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/02-GeneralArrayOperations"
)

add_executable(test_03_2d_ops
    tests/test_03_2d_ops.cpp
    03-2DKernels/MetalOperations.cpp
    03-2DKernels/CPUOperations.cpp
)
configure_metal_target(test_03_2d_ops)
target_include_directories(test_03_2d_ops PRIVATE "${CMAKE_SOURCE_DIR}/03-2DKernels")
add_dependencies(test_03_2d_ops metallib_03)
set_target_properties(test_03_2d_ops PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/03-2DKernels"
)
add_test(NAME test_03_2d_ops COMMAND test_03_2d_ops
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/03-2DKernels"
)

# =============================================================================
# Python bindings (optional — only built when pybind11 is available)
# =============================================================================
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11 found — building Python bindings")
    pybind11_add_module(_m1_gpu_ops python/bindings.cpp)
    target_include_directories(_m1_gpu_ops PRIVATE ${METAL_CPP_DIR})
    target_compile_options(_m1_gpu_ops PRIVATE -fno-objc-arc)
    target_link_libraries(_m1_gpu_ops PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
    )
    # Install the module into the python package directory
    install(TARGETS _m1_gpu_ops DESTINATION m1_gpu_ops)
    install(FILES python/__init__.py DESTINATION m1_gpu_ops)
else()
    message(STATUS "pybind11 not found — skipping Python bindings (install with: pip install pybind11)")
endif()
