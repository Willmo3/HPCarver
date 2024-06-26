# GoogleTest install:
# GoogleTest requires at least C++14
include(FetchContent)
FetchContent_Declare(
googletest
URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# Windows compatibility
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../out/tests)

# Build tests
add_executable(test_serialcarve test_carve.cpp)
target_link_libraries(test_serialcarve GTest::gtest_main)
include(GoogleTest)

add_executable(test_ompcarve test_carve.cpp)
target_link_libraries(test_ompcarve GTest::gtest_main)

add_executable(test_rajacarve test_carve.cpp)
target_link_libraries(test_rajacarve GTest::gtest_main)

if (ENABLE_CUDA)
  add_executable(test_cudacarve test_cudacarve.cpp)
  target_link_libraries(test_cudacarve GTest::gtest_main)
  gtest_discover_tests(test_cudacarve)

  target_link_libraries(test_cudacarve
          cuda_image
          cuda_energy
          cu)
endif ()

gtest_discover_tests(test_serialcarve test_ompcarve test_rajacarve)

# Link all to HPImage
target_link_libraries(test_serialcarve
        hpimage
        energy
        serial)

target_link_libraries(test_ompcarve
        hpimage
        energy
        omp)

target_link_libraries(test_rajacarve
       hpimage
       energy
       raja)