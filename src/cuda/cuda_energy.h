#ifndef HPCARVER_CUDA_ENERGY_H
#define HPCARVER_CUDA_ENERGY_H

// Cuda energy. Overrides regular energy to perform CudaManagedMalloc allocations.
// Author: Will Morris

#include "../energy.h"

namespace hpc_cuda {

class CudaEnergy: carver::Energy {
public:
    /**
     * CudaEnergy constructor.
     * This initializes a block of memory for an energy matrix using cuda
     * @param cols Number of columns to initialize energy block with
     * @param rows Number of rows to initialize energy block with
     */
    CudaEnergy(uint32_t cols, uint32_t rows);

    /**
     * CudaEnergy destructor.
     * Will free cuda shared-allocated resources (i.e. the energy block).
     */
    ~CudaEnergy();
};
}



#endif //HPCARVER_CUDA_ENERGY_H
