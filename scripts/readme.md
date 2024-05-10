# scripts
For ease of use, we've provided several scripts to help configure your HPCarver installs and quickly run tests.

## install
This script installs RAJA the way that this project expects it to be.

You may also build RAJA yourself -- just be sure that a link named "RAJA-install" exists at the root of the HPCarver installation!

### usage
./install [with cuda support (true / anything else)]

### running
Run this script from anywhere within the HPCarver directory!


## build
This script configures CMake to recognize our RAJA install directory and tells it whether to use CUDA.

### usage:
./build [use cuda (true / anything else)]

### running
Run this script from anywhere within the HPCarver directory!


## correct\_tests
This script runs small sanity tests on all implementations.

### options
There are no command line arguments to this script.



## timing\_tests
This script runs timing tests on larger examples. 
It reduces the size of the duke\_dog.ppm image to 600x600, 500x500, and 400x400, tracking the amount of time needed for each implementation.

### usage
./timing_tests [job manager (slurm / anything else)]
