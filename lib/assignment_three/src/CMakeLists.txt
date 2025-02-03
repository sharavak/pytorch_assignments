cmake_minimum_required(VERSION 3.14)

project(cuslogaddexp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(Torch REQUIRED)
add_library(cus_ops cus_ops.cpp)
target_link_libraries(cus_ops "${TORCH_LIBRARIES}")



set_property(TARGET cus_ops PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)