#include "cuda_energy.h"
#include <cassert>

namespace hpc_cuda {

uint32_t *eng_mem(uint32_t size) {
    uint32_t *data;
    assert(cudaMallocManaged(&data, size) == 0);
    return data;
}

CudaEnergy::CudaEnergy(uint32_t cols, uint32_t rows) : carver::Energy(cols, rows, eng_mem) {}

// Cuda destructor
CudaEnergy::~CudaEnergy() {
    // Ensure that no CUDA ops are in progress as we destruct.
    cudaDeviceSynchronize();
    assert(cudaFree(energy) == 0);
    energy = nullptr;
}

// Convert to struct, exposing low-level access to CUDA.
CudaEnergyStruct CudaEnergy::to_struct() {
    return {
        .energy = energy,
        .base_cols = base_cols,
        .current_cols = current_cols,
        .current_rows = current_rows,
    };
}

} // End namespace hpc_cuda
