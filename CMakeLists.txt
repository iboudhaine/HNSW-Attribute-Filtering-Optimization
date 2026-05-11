cmake_minimum_required(VERSION 3.20)
project(anf_hnsw_filter LANGUAGES CXX C VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- third-party (vendored) ----
add_library(croaring STATIC third_party/croaring/roaring.c)
target_include_directories(croaring PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/croaring)
set_property(TARGET croaring PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE ${CMAKE_SOURCE_DIR}/third_party)

# ---- project library (header-only) ----
add_library(anf INTERFACE)
target_include_directories(anf INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(anf INTERFACE hnswlib croaring)

# ---- apps ----
add_executable(benchmark apps/benchmark.cpp)
target_link_libraries(benchmark PRIVATE anf)

add_executable(verify apps/verify.cpp)
target_link_libraries(verify PRIVATE anf)

# ---- tests ----
enable_testing()
add_executable(anf_tests
    tests/test_main.cpp
    tests/test_dataset.cpp
    tests/test_representations.cpp
    tests/test_ground_truth.cpp
    tests/test_filter.cpp)
target_include_directories(anf_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(anf_tests PRIVATE anf)
add_test(NAME anf_tests COMMAND anf_tests)
