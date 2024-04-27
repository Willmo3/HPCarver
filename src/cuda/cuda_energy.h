#ifndef HPCARVER_CUDA_ENERGY_H
#define HPCARVER_CUDA_ENERGY_H

// Cuda energy. Overrides regular energy to perform CudaManagedMalloc allocations.
// Author: Will Morris

#include "../energy.h"

namespace hpc_cuda {

class CudaEnergy: carver::Energy {
protected:
    uint32_t *alloc(uint32_t size) override;
public:

    /**
     * CudaEnergy destructor.
     * Will free cuda shared-allocated resources (i.e. the energy block).
     */
    ~CudaEnergy();
};
}



#endif //HPCARVER_CUDA_ENERGY_H
