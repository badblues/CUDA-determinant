cmake_minimum_required(VERSION 3.21)
project(2 C CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(1 main.c)
add_executable(2 main.cu)

set_target_properties(2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
