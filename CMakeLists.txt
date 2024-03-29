cmake_minimum_required(VERSION 3.5)
project(hpcarver)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../out)

# Prepare ImageMagick
add_definitions( -DMAGICKCORE_QUANTUM_DEPTH=16 )
add_definitions( -DMAGICKCORE_HDRI_ENABLE=0 )
find_package(ImageMagick COMPONENTS Magick++)
include_directories(${ImageMagick_INCLUDE_DIRS})

enable_testing()

# Deferring to cmake
add_subdirectory(src)
add_subdirectory(test)

# TESTING
