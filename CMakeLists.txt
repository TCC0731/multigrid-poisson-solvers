cmake_minimum_required(VERSION 3.20)

project(multigrid_poisson_solvers LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if (ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
    message(WARNING "IPO/LTO is not supported: ${ipo_error}")
endif()

function(enable_release_perf_flags target)
    if (MSVC)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-march=native>
        )
    endif()
endfunction()

add_library(poisson_core
    cpp/src/metrics.cpp
    cpp/src/jacobi.cpp
    cpp/src/operators.cpp
    cpp/src/problem.cpp
    cpp/src/validation.cpp
)

target_include_directories(poisson_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_compile_features(poisson_core PUBLIC cxx_std_20)
enable_release_perf_flags(poisson_core)

if (MSVC)
    target_compile_options(poisson_core PRIVATE /W4)
else()
    target_compile_options(poisson_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(poisson_cpp
    cpp/src/main.cpp
)

target_link_libraries(poisson_cpp PRIVATE poisson_core)
enable_release_perf_flags(poisson_cpp)

if (BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    include(GoogleTest)

    add_executable(poisson_tests
        tests/cpp/test_poisson.cpp
    )

    target_link_libraries(poisson_tests PRIVATE poisson_core GTest::gtest_main)
    target_compile_features(poisson_tests PRIVATE cxx_std_20)
    enable_release_perf_flags(poisson_tests)

    if (MSVC)
        target_compile_options(poisson_tests PRIVATE /W4)
    else()
        target_compile_options(poisson_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    gtest_discover_tests(poisson_tests)
endif()
