#include "cuda_image.h"
#include <cassert>

using namespace hpc_cuda;

hpimage::pixel *CudaImage::alloc(int size) {
    hpimage::pixel *pixel;
    cudaMallocManaged(&pixel, size);
    return pixel;
}

CudaImage::~CudaImage() {
    // Ensure all computation is finished prior to freeing.
    cudaDeviceSynchronize();
    assert(cudaFree(pixels) == 0);
}