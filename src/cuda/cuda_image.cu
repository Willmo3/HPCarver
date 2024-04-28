#include "cuda_image.h"
#include <cassert>

namespace hpc_cuda {
uint32_t *img_mem(uint32_t size) {
    uint32_t *data;
    assert(cudaMallocManaged(&data, size) == 0);
    return data;
}

CudaImage::CudaImage(const char *filename): hpimage::Hpimage(filename, img_mem) {}

CudaImage::~CudaImage() {
    // Ensure all computation is finished prior to freeing.
    cudaDeviceSynchronize();
    assert(cudaFree(pixels) == 0);
}

}