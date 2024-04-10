# Serial execution library
add_library(serial serial.cpp)
target_link_libraries(serial carver)

# Final entry point executable
add_executable(hpcarver hpcarver.cpp)
target_link_libraries(hpcarver
        serial
        hpimage
        RAJA)
