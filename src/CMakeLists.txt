# Serial execution library
add_library(serial serial.cpp)
target_link_libraries(serial carver)

add_library(omp omp.cpp)
target_link_libraries(omp carver OpenMP::OpenMP_CXX)

add_library(pthread pthread.cpp)
target_link_libraries(pthread carver Threads::Threads)

add_library(raja raja.cpp)
target_link_libraries(raja carver RAJA)

# Final entry point executable
add_executable(hpc_serial hpcarver.cpp)
target_link_libraries(hpc_serial
        serial
        hpimage )

add_executable(hpc_omp hpcarver.cpp)
target_link_libraries(hpc_omp
        omp
        hpimage)

add_executable(hpc_pthread hpcarver.cpp)
target_link_libraries(hpc_pthread
        pthread
        hpimage)

add_executable(hpc_raja hpcarver.cpp)
target_link_libraries(hpc_raja
        raja
        hpimage)
