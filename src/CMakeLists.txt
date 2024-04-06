# Serial execution library
add_library(serial serial/serial.cpp serial/serial.h)
target_link_libraries(serial hpimage)

# Final entry point executable
add_executable(hpcarver hpcarver.cpp)
target_link_libraries(hpcarver
        serial
        hpimage
        RAJA)
