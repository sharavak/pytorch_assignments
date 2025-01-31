cmake_minimum_required(VERSION 3.14)
project(test_torch_logaddexp)
enable_testing()
set(GTEST_ROOT "../tests/googletest")
 
#set(Torch_DIR "/home/shar/libtorch/share/cmake/Torch")
#list(APPEND CMAKE_PREFIX_PATH "/home/shar/libtorch/share/cmake/Torch")
 
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)
 
add_subdirectory(../tests/googletest googletest-build)
 
 
add_executable(test_cusops test_cusops.cpp ../src/cus_ops.cpp)
 
target_link_libraries(test_cusops ${TORCH_LIBRARIES} gtest gtest_main pthread )
 
target_sources(test_cusops PRIVATE ../tests/googletest/googletest/src/gtest-all.cc)
target_include_directories(test_cusops PUBLIC ${GTEST_INCLUDE_DIRS})
 
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
 
add_test(NAME OpsTest COMMAND test_cusops)
