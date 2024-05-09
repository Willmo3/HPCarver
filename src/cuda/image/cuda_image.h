#ifndef HPCARVER_CUDA_IMAGE_H
#define HPCARVER_CUDA_IMAGE_H

/**
 * CUDA image. Memory allocated using CUDA functions.
 * This is a variant of the HPImage library.
 *
 * Author: Will Morris
 */

#include "../../../HPImage/hpimage.h"

namespace hpc_cuda {

// CUDA functions cannot call standard image helpers.
// This means they need lower-level access to the image data.
struct CudaImageStruct {
    hpimage::pixel *pixels;
    size_t base_cols;
    size_t current_cols;
    size_t current_rows;
};

class CudaImage: public hpimage::Hpimage {

public:
    CudaImage(const char *filename);
    ~CudaImage() override;

    /**
     * Convert the fields of this CudaImage to a struct.
     * Allowing direct access to the internal memory layout.
     * @return The struct.
     */
    CudaImageStruct to_struct();
};

} // hpc_cuda

#endif //HPCARVER_CUDA_IMAGE_H
