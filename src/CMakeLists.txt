# Energy matrix library
add_library(energy common/carver/energy.cpp common/carver/energy.h)

# Carver common library
add_library(carver common/carver/carver.cpp common/carver/carver.h)
target_link_libraries(carver hpimage energy)

# Timing utilities
add_library(timer common/timer.cpp common/timer.h)

# Serial execution library
add_library(serial serial/carver/serial_carver.cpp)
target_link_libraries(serial carver)

add_library(omp omp/carver/omp_carver.cpp)
target_link_libraries(omp carver OpenMP::OpenMP_CXX)

add_library(raja raja/carver/raja_carver.cpp)
target_link_libraries(raja carver RAJA)

# Add CUDA versions of libraries
# Good news: should be enough to just subclass
if (ENABLE_CUDA)
    add_library(cuda_energy cuda/carver/cuda_energy.cu cuda/carver/cuda_energy.h)
    set_target_properties(cuda_energy PROPERTIES CUDA_ARCHITECTURES "all-major")
    target_link_libraries(cuda_energy energy)

    add_library(cuda_image cuda/image/cuda_image.h cuda/image/cuda_image.cu)
    set_target_properties(cuda_image PROPERTIES CUDA_ARCHITECTURES "all-major")
    target_link_libraries(cuda_image hpimage)

    add_library(cu cuda/carver/cuda_carver.cu)
    set_target_properties(cu PROPERTIES CUDA_ARCHITECTURES "all-major")
    target_link_libraries(cu carver cuda_energy)

    add_executable(hpc_cuda cuda/cuda_main.cpp)
    set_target_properties(hpc_cuda PROPERTIES CUDA_ARCHITECTURES "all-major")
    target_link_libraries(hpc_cuda
                            timer
                            cu
                            cuda_energy
                            cuda_image)
endif ()

# Final entry point executable
add_executable(hpc_serial serial/serial_main.cpp)
target_link_libraries(hpc_serial
        serial
        timer
        hpimage)

add_executable(hpc_omp omp/omp_main.cpp)
target_link_libraries(hpc_omp
        omp
        timer
        hpimage)

add_executable(hpc_raja raja/raja_main.cpp)
target_link_libraries(hpc_raja
        raja
        timer
        hpimage)

# Tests have their own CMakeLists
add_subdirectory(test)


