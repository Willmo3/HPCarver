#include "cuda_image.h"
#include <cassert>

namespace hpc_cuda {

hpimage::pixel *img_mem(uint32_t size) {
    hpimage::pixel *data;
    assert(cudaMallocManaged(&data, size) == 0);
    return data;
}

CudaImage::CudaImage(const char *filename): hpimage::Hpimage(filename, img_mem) {}

CudaImage::~CudaImage() {
    // Ensure all computation is finished prior to freeing.
    cudaDeviceSynchronize();
    assert(cudaFree(pixels) == 0);
    pixels = nullptr;
}

CudaImageStruct CudaImage::to_struct() {
    return {
        .pixels = pixels,
        .base_cols = base_cols,
        .current_cols = current_cols,
        .current_rows = current_rows,
    };
}

} // hpc_cuda
