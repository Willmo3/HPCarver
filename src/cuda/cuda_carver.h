//
// Created by morri2wj on 4/28/24.
//

#ifndef HPCARVER_CUDA_CARVER_H
#define HPCARVER_CUDA_CARVER_H

#include "../carver.h"
#include "cuda_image.h"
#include "cuda_energy.h"

namespace hpc_cuda {

/**
 * CudaCarver. Shadows carver image and energy matrix fields.
 *
 * Objective:
 */
class CudaCarver: carver::Carver {
private:
    // Image and
    CudaImage &image;
    CudaEnergy energy;

public:
    /**
     * CudaCarver constructor.
     * NOTE: while the API is external, its internal policies are implementation specific.
     * I.E. the pthreads version may need to initialize a thread pool.
     *
     * @param image Image to operate on.
     */
    explicit CudaCarver(CudaImage &image);

    /**
     * ACCESSORS
     */
    hpimage::Hpimage *get_image() override;
    carver::Energy *get_energy() override;
};

} // end hpc cuda

#endif //HPCARVER_CUDA_CARVER_H
