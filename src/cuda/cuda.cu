#include "../carver.h"
#include "cuda_energy.h"

using namespace carver;

Carver::Carver(hpimage::Hpimage &image): image(image), energy(hpc_cuda::CudaEnergy(image.cols(), image.rows()))
{ assert_valid_dims(); }