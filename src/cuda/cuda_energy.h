#ifndef HPCARVER_CUDA_ENERGY_H
#define HPCARVER_CUDA_ENERGY_H

// Cuda energy. Overrides regular energy to perform CudaManagedMalloc allocations.
// Author: Will Morris

#include "../carver/energy.h"

namespace hpc_cuda {

class CudaEnergy: public carver::Energy {
public:
    /**
     * CudaEnergy constructor. Calls CUDA allocator for cols * rows memory.
     * @param cols Number of columns for energy matrix.
     * @param rows Number of rows for energy matrix.
     */
    CudaEnergy(uint32_t cols, uint32_t rows);

    /**
     * CudaEnergy destructor.
     * Will free cuda shared-allocated resources (i.e. the energy block).
     */
    ~CudaEnergy() override;

    /**
     * For CUDA functions, the strict memory protections in the API must be loosened.
     * We need direct access to the energy matrix!
     * @return The energy matrix associated with this object.
     */
    uint32_t *get_energy_matrix();
};
} // hpc_cuda



#endif //HPCARVER_CUDA_ENERGY_H
