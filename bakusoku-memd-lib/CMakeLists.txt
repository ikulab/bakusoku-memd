cmake_minimum_required(VERSION 3.16)
project(bakusoku-memd-lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if ((NOT UNIX) AND (NOT APPLE))
    message(WARNING "このシステム:${CMAKE_SYSTEM_NAME} ではテストしてないので何が起こるかわからない!")
endif ()

add_library(bakusoku-memd-lib INTERFACE)
target_include_directories(bakusoku-memd-lib SYSTEM INTERFACE third-party/xtl/include)
target_include_directories(bakusoku-memd-lib SYSTEM INTERFACE third-party/xtensor/include)
target_include_directories(bakusoku-memd-lib SYSTEM INTERFACE third-party/xtensor-blas/include)
target_include_directories(bakusoku-memd-lib INTERFACE include)

target_compile_options(bakusoku-memd-lib INTERFACE
        $<$<CXX_COMPILER_ID:AppleClang>:-v -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-documentation -Wno-shadow-field-in-constructor -Wno-shadow -Wno-missing-prototypes>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic>
        $<$<CXX_COMPILER_ID:MSVC>: /W4>
        $<$<CXX_COMPILER_ID:GNU>: -Wall>)

include(local-properties-default.cmake)
include(local-properties.cmake OPTIONAL)
set_local_properties()

if (UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(bakusoku-memd-lib INTERFACE Threads::Threads)
endif ()

find_package(OpenBLAS REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(bakusoku-memd-lib INTERFACE ${OpenBLAS_LIBRARIES} ${MPI_LIBRARIES})
target_include_directories(bakusoku-memd-lib SYSTEM INTERFACE ${MPI_INCLUDE_PATH})
target_compile_options(bakusoku-memd-lib INTERFACE ${OpenMP_CXX_FLAGS})
target_link_options(bakusoku-memd-lib INTERFACE ${OpenMP_CXX_FLAGS})


enable_testing()
option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
option(INSTALL_GTEST "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" OFF)
add_subdirectory(third-party/googletest)
include(GoogleTest)
add_executable(bakusoku-memd-lib-test test/main.cpp)
target_link_libraries(bakusoku-memd-lib-test bakusoku-memd-lib gtest gtest_main)
add_custom_command(TARGET bakusoku-memd-lib-test POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/test/res $<TARGET_FILE_DIR:bakusoku-memd-lib-test>/res)
gtest_discover_tests(bakusoku-memd-lib-test)
