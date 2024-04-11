# Serial execution library
add_library(serial serial.cpp)
target_link_libraries(serial carver)

add_library(omp omp.cpp)
target_link_libraries(omp carver OpenMP::OpenMP_CXX)

# Final entry point executable
add_executable(hpc_serial hpcarver.cpp)
target_link_libraries(hpc_serial
        serial
        hpimage
        RAJA)

add_executable(hpc_omp hpcarver.cpp)
target_link_libraries(hpc_omp
        omp
        hpimage
        RAJA)
