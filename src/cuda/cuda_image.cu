#include "cuda_image.h"

using namespace hpc_cuda;

hpimage::pixel *CudaImage::alloc(int size) {
    hpimage::pixel *pixel;
    cudaMallocManaged(&pixel, size);
    return pixel;
}

CudaImage::~CudaImage() {
    cudaFree(pixels);
}