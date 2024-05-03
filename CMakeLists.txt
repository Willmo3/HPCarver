cmake_minimum_required(VERSION 3.5)
project(hpcarver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For now, all of these dependencies are required.
find_package(RAJA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Done because of nested cmake structure
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../out)

# HPImage library
add_library(hpimage HPImage/hpimage.cpp HPImage/hpimage.h)

# Energy matrix library
add_library(energy src/carver/energy.cpp src/carver/energy.h)

# Carver common library
add_library(carver src/carver/carver.cpp src/carver/carver.h)
target_link_libraries(carver hpimage energy)

# Timing utilities
add_library(timer src/carver/timer.cpp src/carver/timer.h)

# Deferring to cmake
add_subdirectory(src)
add_subdirectory(test)

# For right now, ignore tests -- see if it fixes the issue!
