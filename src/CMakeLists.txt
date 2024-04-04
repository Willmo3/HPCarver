# Add hpimage library -- base
add_library(hpimage hpimage/hpimage.cpp hpimage/hpimage.h)

# Serial execution library
add_library(serial serial/serial.cpp serial/serial.h)

# Final entry point executable
add_executable(hpcarver hpcarver.cpp)
target_link_libraries(hpcarver
        ${ImageMagick_LIBRARIES}
        hpimage
        serial
        RAJA)
