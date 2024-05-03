# High Performance Carver (HPC)

## Introduction
High Performance Carver is an implementation of the seam carving algorithm for content-aware image resizing.
Its purpose is not to be used as a production application, but rather, to test the RAJA performance portability suite against a number of existing local parallelization alternatives.

## Credits
1) Dr. Chris Mayfield's CS 159 Seam Carving PA
2) Princeton University's seam carving programming assignment:
   (https://www.cs.princeton.edu/courses/archive/fall14/cos226/assignments/seamCarving.html)
3) The original seam carving implementation, by Advan and Shamir
   (https://dl.acm.org/doi/10.1145/1276377.1276390)

## Dependencies

### Prerequisites
These dependencies should be installed on your system prior to installing this project.

1. c++ build essentials (makefile, git, g++)
2. CMake

### Pulled
These dependencies are pulled when the project is installed.

1. RAJA (you can symlink to the top level of your own RAJA-install directory if you prefer)
2. HPImage (make sure to clone with git clone --recurse-submodules to get this!)

Refer to the installation section for more details!

### HPImage
This project requires the HPImage library for processing. Please do *git clone --recursive* to ensure that HPImage is cloned as well! If the version of HPImage provided is out of date, git submodule update should fix this.

Additionally, you can move a separate version of HPImage into the root of the project. You should not need to build HPImage by itself -- the provided .h and .cpp file are aadequate when properly configured.

## Installation
We provide several install scripts for this project. 
If you are building in a non-cuda ecosystem, please use the "nocuda" or "noncuda" versions of these scripts.
All scripts can be found in the scripts directory at the root of the project.

1. Install RAJA using scripts/install
2. Build the project using scripts/build

## Timing
Run ./timing to see timing results!

The structure of our timing tests will be described more extensively in the final deliverable. 
We run three trials for each test on duke_dog.ppm, carving them down to progressively smaller images (600x600, 500x500, 400x400).

### Correctness
Run ./correct_tests to quickly evaluate whether the solutions are working!

I've copied simple examples from the Princeton programming assignment for basic energy. 

We will be using the Google Testing framework here. 
It's designed for cpp (which will be good for RAJA!) but also works for C. 
See https://google.github.io/googletest/primer.html.

## Design notes

### Energy function
We employ the dual-gradient energy function described in https://www.cs.princeton.edu/courses/archive/fall14/cos226/assignments/seamCarving.html.

### Minimum seam detection.
As the matrix representation of images indicates, this problem can be modeled as a graph traversal.

Let each pixel *p* be a vertex in the graph. 

For each *p*, weight *p*'s incoming edges by its energy.

The minimum energy path through the graph is the path from top to bottom consisting of the lowest total weight between all edges.

Given these basic energy weights, we could repeatedly compute the optimal seam using an all-source-shortest-path algorithm. But it turns out there's an easier way!

Suppose we're computing a vertical seam from the top of the image to the bottom. Then each vertex has three incoming edges, since images are rectangular: the three pixels on top of it. (We wrap on edges).
Or the pixel has no incoming edges, because it's on top of the image.

Since all of these edges are weighted by energy, the minimum energy of a pixel with regard to a seam is not only the basic energy of its dual gradient,
but also the cumulative minimum energy of the path it took to reach it.

The minimum energy to reach this pixel is simply the minimum energy of its upper neighbors, or zero if it's a root node.

Observe that this recursive definition of a pixel's energy can easily be computed via dynamic programming. 
Simply encode each pixel's energy as the sum of its dual gradient energy and the minimum energies of all its incoming edges.

Then the optimal seam can be computed by traversing from the bottom of the modified energy matrix to the top. 
Any seam of minimal total energy at the bottom of the matrix will be a minimum energy path! 

This approach is described in the original seam carving document, https://dl.acm.org/doi/10.1145/1276377.1276390.

        
