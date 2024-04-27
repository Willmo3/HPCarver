#include "cuda_energy.h"
#include <cassert>

using namespace hpc_cuda;

// Cuda constructor
uint32_t *CudaEnergy::alloc(uint32_t size) {
    uint32_t *data;
    assert(cudaMallocManaged(&data, size) == 0);
    return data;
}

// Cuda destructor
CudaEnergy::~CudaEnergy() {
    // Ensure that no CUDA ops are in progress as we destruct.
    cudaDeviceSynchronize();
    assert(cudaFree(&energy) == 0);
}
