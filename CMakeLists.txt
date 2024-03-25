cmake_minimum_required(VERSION 3.5)
project(hpcarver)

set(SOURCE_FILES hpcarver.cpp)
add_executable(hpcarver ${SOURCE_FILES})

# Link to ImageMagick
add_definitions( -DMAGICKCORE_QUANTUM_DEPTH=16 )
add_definitions( -DMAGICKCORE_HDRI_ENABLE=0 )
find_package(ImageMagick COMPONENTS Magick++)
include_directories(${ImageMagick_INCLUDE_DIRS})
target_link_libraries(hpcarver ${ImageMagick_LIBRARIES})
