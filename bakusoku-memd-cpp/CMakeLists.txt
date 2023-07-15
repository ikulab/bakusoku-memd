cmake_minimum_required(VERSION 3.16)
project(bakusoku-memd-cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(bakusoku-memd src/main.cpp)

target_link_libraries(bakusoku-memd PRIVATE bakusoku-memd-lib)

target_include_directories(bakusoku-memd PRIVATE)
