cmake_minimum_required(VERSION 3.5)
project(hpcarver)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../out)

# GoogleTest install:
# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Prepare ImageMagick
add_definitions( -DMAGICKCORE_QUANTUM_DEPTH=16 )
add_definitions( -DMAGICKCORE_HDRI_ENABLE=0 )
find_package(ImageMagick COMPONENTS Magick++)
include_directories(${ImageMagick_INCLUDE_DIRS})

# Build tests
enable_testing()
add_executable(test_imagemagick test/test_imagemagick.cpp)
target_link_libraries(test_imagemagick GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_imagemagick)

# Build carver
add_executable(hpcarver src/hpcarver.cpp)

# Link all to ImageMagick
target_link_libraries(hpcarver ${ImageMagick_LIBRARIES})
target_link_libraries(test_imagemagick ${ImageMagick_LIBRARIES})
