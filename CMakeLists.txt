cmake_minimum_required(VERSION 3.5)
project(hpcarver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For now, all of these dependencies are required.
find_package(RAJA REQUIRED)
find_package(OpenMP REQUIRED)

# HPImage library
add_library(hpimage HPImage/hpimage.cpp HPImage/hpimage.h)

# Done because of nested cmake structure
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../out)

add_subdirectory(src)