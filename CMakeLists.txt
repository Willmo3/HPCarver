cmake_minimum_required(VERSION 3.5)
project(hpcarver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(RAJA REQUIRED)

# Done because of nested cmake structure
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../out)

# Prepare HPImage
add_library(hpimage HPImage/hpimage.cpp HPImage/hpimage.h)

# Deferring to cmake
add_subdirectory(src)
add_subdirectory(test)

# For right now, ignore tests -- see if it fixes the issue!
