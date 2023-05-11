cmake_minimum_required(VERSION 3.10)
project(tensor)

add_subdirectory(external/fmt)
add_subdirectory(external/pybind11)
add_subdirectory(external/googletest)

enable_testing()
add_executable(tensor_test cpp/tensor_test.cpp cpp/tensor.cpp cpp/tensor.hpp)
target_link_libraries(tensor_test gtest_main fmt)
add_test(NAME tensor_test COMMAND tensor_test)

set(CMAKE_CXX_FLAGS "-openmp:experimental")
find_package(OpenMP REQUIRED)
pybind11_add_module(tensor cpp/tensor.cpp cpp/tensor.hpp cpp_wrapper/tensor_wrapper.cpp)
target_link_libraries(tensor PUBLIC fmt OpenMP::OpenMP_CXX)