cmake_minimum_required(VERSION 3.25)
project(meridian VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(MERIDIAN_BUILD_TESTS "Build unit and fuzz tests" ON)
option(MERIDIAN_BUILD_BENCH "Build latency benchmarks" ON)
option(MERIDIAN_NATIVE "Tune for the build machine (-march=native)" ON)
option(MERIDIAN_SANITIZE "Instrument with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

# ---- header-only engine core -------------------------------------------------------------
add_library(meridian INTERFACE)
add_library(meridian::meridian ALIAS meridian)
target_include_directories(meridian INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(meridian INTERFACE cxx_std_20)
target_compile_options(meridian INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wno-c++17-extensions)
if(MERIDIAN_NATIVE AND NOT MERIDIAN_SANITIZE)
  target_compile_options(meridian INTERFACE -march=native)
endif()
if(MERIDIAN_SANITIZE)
  target_compile_options(meridian INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(meridian INTERFACE -fsanitize=address,undefined)
else()
  # Self-contained binaries: no dependency on the toolchain's libc++/libunwind DLLs.
  target_link_options(meridian INTERFACE -static)
endif()

# ---- tools ---------------------------------------------------------------------------------
add_executable(meridian_replay tools/replay.cpp)
target_link_libraries(meridian_replay PRIVATE meridian)

# ---- benchmarks ----------------------------------------------------------------------------
if(MERIDIAN_BUILD_BENCH)
  add_executable(meridian_bench bench/bench_engine.cpp)
  target_link_libraries(meridian_bench PRIVATE meridian)
  set_target_properties(meridian_bench PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# ---- tests ---------------------------------------------------------------------------------
if(MERIDIAN_BUILD_TESTS)
  include(FetchContent)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(meridian_tests
    tests/test_bitset.cpp
    tests/test_order_book.cpp
    tests/test_risk.cpp
    tests/test_wire.cpp
    tests/test_fuzz.cpp)
  target_link_libraries(meridian_tests PRIVATE meridian GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(meridian_tests DISCOVERY_MODE PRE_TEST)
endif()
