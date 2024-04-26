#include "cuda_energy.h"
#include <cassert>

using namespace hpc_cuda;

// Cuda constructor
CudaEnergy::CudaEnergy(uint32_t cols, uint32_t rows) {
    base_cols = cols;
    base_rows = rows;
    current_cols = cols;
    current_rows = rows;

    assert(cudaMallocManaged(&energy, sizeof(uint32_t) * rows * cols) == 0);
}

// Cuda destructor
CudaEnergy::~CudaEnergy() {
    // Ensure that no CUDA ops are in progress as we destruct.
    cudaDeviceSynchronize();
    assert(cudaFree(&energy) == 0);
}
