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
    cpp/src/gs.cpp
    cpp/src/jacobi.cpp
    cpp/src/mg.cpp
    cpp/src/red_black.cpp
    cpp/src/sor.cpp
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

add_executable(poisson_benchmark_cpp
    benchmark/main.cpp
)

target_link_libraries(poisson_benchmark_cpp PRIVATE poisson_core)
target_compile_definitions(poisson_benchmark_cpp PRIVATE POISSON_BENCHMARK_BACKEND_LABEL=cpp20)
enable_release_perf_flags(poisson_benchmark_cpp)

find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    add_library(poisson_omp_core
        cpp_omp/src/metrics.cpp
        cpp_omp/src/gs.cpp
        cpp_omp/src/jacobi.cpp
        cpp_omp/src/mg.cpp
        cpp_omp/src/red_black.cpp
        cpp_omp/src/sor.cpp
        cpp_omp/src/operators.cpp
        cpp_omp/src/problem.cpp
        cpp_omp/src/validation.cpp
    )

    target_include_directories(poisson_omp_core
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp_omp/include
    )

    target_compile_features(poisson_omp_core PUBLIC cxx_std_20)
    target_link_libraries(poisson_omp_core PUBLIC OpenMP::OpenMP_CXX)
    enable_release_perf_flags(poisson_omp_core)

    if (MSVC)
        target_compile_options(poisson_omp_core PRIVATE /W4)
    else()
        target_compile_options(poisson_omp_core PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(poisson_cpp_omp
        cpp_omp/src/main.cpp
    )

    target_link_libraries(poisson_cpp_omp PRIVATE poisson_omp_core)
    enable_release_perf_flags(poisson_cpp_omp)

    add_executable(poisson_benchmark_omp
        benchmark/main.cpp
    )

    target_link_libraries(poisson_benchmark_omp PRIVATE poisson_omp_core)
    target_compile_definitions(poisson_benchmark_omp PRIVATE POISSON_BENCHMARK_BACKEND_LABEL=cpp20_omp)
    enable_release_perf_flags(poisson_benchmark_omp)
else()
    message(WARNING "OpenMP not found: skipping poisson_cpp_omp and poisson_benchmark_omp targets")
endif()

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

    if (OpenMP_CXX_FOUND)
        add_executable(poisson_tests_omp
            tests/cpp/test_poisson_omp.cpp
        )

        target_link_libraries(poisson_tests_omp PRIVATE poisson_omp_core GTest::gtest_main)
        target_compile_features(poisson_tests_omp PRIVATE cxx_std_20)
        enable_release_perf_flags(poisson_tests_omp)

        if (MSVC)
            target_compile_options(poisson_tests_omp PRIVATE /W4)
        else()
            target_compile_options(poisson_tests_omp PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        gtest_discover_tests(poisson_tests_omp TEST_PREFIX omp.)
    endif()
endif()
