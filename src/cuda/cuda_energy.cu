#include "cuda_energy.h"
#include <cassert>
#include <iostream>

namespace hpc_cuda {

uint32_t *mem(uint32_t size) {
    uint32_t *data;
    assert(cudaMallocManaged(&data, size) == 0);
    return data;
}

CudaEnergy::CudaEnergy(uint32_t cols, uint32_t rows) : carver::Energy(cols, rows, mem) {}

// Cuda destructor
CudaEnergy::~CudaEnergy() {
    std::cout << "dealloc cuda" << std::endl;
    // Ensure that no CUDA ops are in progress as we destruct.
    cudaDeviceSynchronize();
    assert(cudaFree(&energy) == 0);
}

} // End namespace hpc_cuda
