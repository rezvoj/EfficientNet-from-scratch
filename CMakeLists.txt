# ===================================================================
# 1. Basic Project Setup (Required for all CMake projects)
# ===================================================================

# Set the minimum required version of CMake. 3.16 is a good modern choice.
cmake_minimum_required(VERSION 3.16)

# Define the project name and specify the programming languages used.
# Since you have .cu files, we must include CUDA.
project(EfficientNetFromScratch LANGUAGES CXX CUDA)

# Set the C++ standard to C++17 for modern features like std::filesystem.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- NEW CODE BLOCK TO FIX THE WARNING ---
# Set the CUDA architectures to build for. Find your GPU's "Compute Capability"
# from NVIDIA's website and set the value below (without the dot).
# Examples: Turing (RTX 20xx) -> 75, Ampere (RTX 30xx) -> 86, Ada (RTX 40xx) -> 89
set(CMAKE_CUDA_ARCHITECTURES 89) # <-- EDIT THIS VALUE FOR YOUR GPU
if(MSVC)
  # This forces the use of the static runtime library (/MT) for Release builds.
  # This must match how OpenCV was compiled to avoid linker errors.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# ===================================================================
# 2. Find Dependencies (Like OpenCV)
# ===================================================================

# This block finds your custom-built OpenCV.
# The path must point to the directory containing "OpenCVConfig.cmake".
# This is the path we decided on. If you installed it elsewhere, change this line.
set(OpenCV_DIR "C:/dev/opencv_install/staticlib")
find_package(OpenCV REQUIRED)

# This message is just a sanity check to confirm it found OpenCV.
message(STATUS "Found OpenCV version ${OpenCV_VERSION} in ${OpenCV_DIR}")


# ===================================================================
# 3. Define Your Main Project Target(s)
# ===================================================================

# Use file(GLOB_RECURSE...) to automatically find all .cpp and .cu source files in the src/ directory.
file(GLOB_RECURSE EFFICIENTNET_SOURCES "src/*.cpp" "src/*.cu")

# Create a library from all your source files.
add_library(efficientnet_lib ${EFFICIENTNET_SOURCES})

# Link your library against OpenCV.
# FIX: Use PUBLIC so that any target linking to efficientnet_lib (like main_app)
# also inherits the include directories from OpenCV.
target_link_libraries(efficientnet_lib PUBLIC ${OpenCV_LIBS})

# We still need a main executable to run the project.
add_executable(main_app src/main.cpp)

# Link the main executable against your library.
target_link_libraries(main_app PRIVATE efficientnet_lib)


# ===================================================================
# 4. Define Tools (Our Converter)
# ===================================================================

# Add an executable target for our converter tool.
add_executable(cifar10_converter tools/cifar10_converter.cpp)

# Since the tool uses std::filesystem, we must enable C++17 (already done above).

# Link the converter against the OpenCV libraries we found.
target_link_libraries(cifar10_converter PRIVATE ${OpenCV_LIBS})

# Set the output directory for the tool's executable
set_target_properties(cifar10_converter PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)