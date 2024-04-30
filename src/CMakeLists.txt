# Serial execution library
add_library(serial serial/serial.cpp)
target_link_libraries(serial carver)

add_library(omp omp/omp.cpp)
target_link_libraries(omp carver OpenMP::OpenMP_CXX)

add_library(pthread pthread/pthread.cpp)
target_link_libraries(pthread carver Threads::Threads)

add_library(raja raja/raja.cpp)
target_link_libraries(raja carver RAJA)

# Add CUDA versions of libraries
# Good news: should be enough to just subclass
if (ENABLE_CUDA)
    add_library(cuda_energy cuda/cuda_energy.cu cuda/cuda_energy.h)
    set_target_properties(cuda_energy PROPERTIES CUDA_ARCHITECTURES "all-major")
    target_link_libraries(cuda_energy energy)

    add_library(cuda_image cuda/cuda_image.h cuda/cuda_image.cu)
    set_target_properties(cuda_image PROPERTIES CUDA_ARCHITECTURES "all-major")
    target_link_libraries(cuda_image hpimage)

    add_library(cu cuda/cuda.cu)
    set_target_properties(cu PROPERTIES CUDA_ARCHITECTURES "all-major")
    target_link_libraries(cu carver)

    add_executable(hpc_cuda cuda/hpc_cuda.cpp)
    set_target_properties(hpc_cuda PROPERTIES CUDA_ARCHITECTURES "all-major")
    target_link_libraries(hpc_cuda
                            cu
                            cuda_energy
                            cuda_image)
endif ()

# Final entry point executable
add_executable(hpc_serial serial/hpc_serial.cpp)
target_link_libraries(hpc_serial
        serial
        hpimage)

add_executable(hpc_omp omp/hpc_omp.cpp)
target_link_libraries(hpc_omp
        omp
        hpimage)

add_executable(hpc_pthread pthread/hpc_pthread.cpp)
target_link_libraries(hpc_pthread
        pthread
        hpimage)

add_executable(hpc_raja raja/hpc_raja.cpp)
target_link_libraries(hpc_raja
        raja
        hpimage)


