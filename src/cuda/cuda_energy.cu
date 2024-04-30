#include "cuda_energy.h"
#include <cassert>

namespace hpc_cuda {

uint32_t *mem(uint32_t size) {
    uint32_t *data;
    assert(cudaMallocManaged(&data, size) == 0);
    return data;
}

CudaEnergy::CudaEnergy(uint32_t cols, uint32_t rows) : carver::Energy(cols, rows, mem) {}

// Cuda destructor
CudaEnergy::~CudaEnergy() {
    // Ensure that no CUDA ops are in progress as we destruct.
    cudaDeviceSynchronize();
    assert(cudaFree(energy) == 0);
    energy = nullptr;
}

uint32_t *CudaEnergy::get_energy_matrix() {
    return energy;
}

} // End namespace hpc_cuda
