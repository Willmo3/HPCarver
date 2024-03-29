# Add hpimage library
# Treating as library to ease linking with testing.
add_library(hpimage hpimage/hpimage.cpp hpimage/hpimage.h)

add_executable(hpcarver hpcarver.cpp)
# Link carver to ImageMagick and hpimage.
target_link_libraries(hpcarver
        ${ImageMagick_LIBRARIES}
        hpimage)