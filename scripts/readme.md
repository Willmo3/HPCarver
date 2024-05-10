# Configuration scripts

For ease of use, we've provided several scripts to help configure your RAJA installs.

## Install
This script installs RAJA the way that this project expects it to be.

You may also build RAJA yourself -- just be sure that a link named "RAJA-install" exists at the root of the HPCarver installation!

### Options
There is one argument to this script: whether to install RAJA with CUDA support. Type in "true" for yes, anything else (or nothing!) for no.

### Running
Run this script from anywhere within the HPCarver directory!


## Build
This script configures CMake to recognize our RAJA install directory and tells it whether to use CUDA.

### Options
Again, there's just one argument to this script: whether to use CUDA! Pass it "true" for yes, anything else for no.
