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

## Installation
Install RAJA, building it with support for all desired backends.
Please refer to the install script for some relevant options (https://github.com/470-research-crew/install) and the RAJA user guide (https://raja.readthedocs.io/en/develop/sphinx/user_guide/index.html).

The main insight is that RAJA must be built with support for all desired backends, such as OpenMP. At this stage, we've built it with support for OpenMP and CUDA. Since we're targeting lower-performance end user machines, not supporting MPI is not a problem.

Once you've done that, specify your RAJA install directory in the makefile.

## Working Notes
### RAJA Setup
Setting up RAJA has been a big ordeal!

Firstly, it's important to note that RAJA should not just be made, it should be *make installed*, with a path for an installation directory provided. Simply making does not appear to be enough!

Crucially, do not allow RAJA to use the default make install directory! This will likely require superuser perms. 
I ran cmake -DCMAKE_INSTALL_PREFIX=/nfs/home/morri2wj/applications/raja ../ within the build directory to tell RAJA where to install itself when make install is called. You must do this before you even make!

Then, you will be able to link with RAJA using the raja-config.cmake files in that library directory.
Be sure to pass the path to this library to cmake, NOT what the RAJA people suggest! As far as I can tell, this works.

In my case, I built RAJA to ~/applications/raja, so when I built hpcarver I passed -DRAJADIR="~/applications/raja/lib/cmake/raja". I have not tested this extensively, but it was at least able to find the headers.

#### RAJA Dependencies
For the most part, RAJA appears to install its own dependencies, such as the blt build system and the camp metaprogramming library (both from Lawrence Livermore).
If we need these for other purposes, we would need to configure RAJA to use our own installed version. I doubt this will be a problem.

### Testing
I've copied simple examples from the Princeton programming assignment for basic energy. 

To create simple images, I've used the pixilart online editor.

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

        
