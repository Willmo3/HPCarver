#ifndef HPCARVER_CUDA_IMAGE_H
#define HPCARVER_CUDA_IMAGE_H

/**
 * CUDA image. Memory allocated using CUDA functions.
 * This is a variant of the HPImage library.
 *
 * Author: Will Morris
 */

#include "../../HPImage/hpimage.h"

namespace hpc_cuda {

class CudaImage: hpimage::Hpimage {

protected:
    hpimage::pixel *alloc(uint32_t size) override;
public:
    ~CudaImage();
};

}

#endif //HPCARVER_CUDA_IMAGE_H
