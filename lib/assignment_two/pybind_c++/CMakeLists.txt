cmake_minimum_required(VERSION 3.28.3)
project(pybind_ops)


set(Torch_DIR "/home/shar/libtorch/share/cmake/Torch")
list(APPEND CMAKE_PREFIX_PATH "/home/shar/libtorch/share/cmake/Torch")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})


file (GLOB SOURCE_FILES "cus_min_wrap.cpp")
file (GLOB HEADER_FILES "/home/shar/libtorch/share/cmake/Torch" "../src/cus_min.h")


pybind11_add_module(custom_min ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(custom_min PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(custom_min PUBLIC pybind11::module ${TORCH_LIBRARIES})
