cmake_minimum_required(VERSION 3.16)

project(bakusoku-memd)

enable_testing()

add_subdirectory(bakusoku-memd-lib)

add_subdirectory(bakusoku-memd-cpp)

add_executable(bakusoku-memd-cpp IMPORTED)