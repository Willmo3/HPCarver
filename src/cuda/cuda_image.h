/**
 * CUDA image. Memory allocated using CUDA functions.
 * This is a variant of the HPImage library.
 *
 * Author: Will Morris
 */
#ifndef HPCARVER_CUDA_IMAGE_H
#define HPCARVER_CUDA_IMAGE_H

#include "../../HPImage/hpimage.h"

namespace hpc_cuda {

class CudaImage: hpimage::Hpimage {
public:
    /**
     * CudaEnergy constructor.
     * This initializes a block of memory for an energy matrix using cuda
     * @param cols Number of columns to initialize energy block with
     * @param rows Number of rows to initialize energy block with
     */
    CudaImage(uint32_t cols, uint32_t rows);

    /**
     * CudaEnergy destructor.
     * Will free cuda shared-allocated resources (i.e. the energy block).
     */
    ~CudaImage();
};
}

#endif //HPCARVER_CUDA_IMAGE_H
