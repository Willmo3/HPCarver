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
}

} // hpc_cuda
