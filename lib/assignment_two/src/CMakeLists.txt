cmake_minimum_required(VERSION 3.14)

project(cusmin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(Torch REQUIRED)
add_library(cus_min cus_min.cpp)
target_link_libraries(cus_min "${TORCH_LIBRARIES}")



set_property(TARGET cus_min PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)