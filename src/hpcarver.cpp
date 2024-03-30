// Entry file for HPcarver
// Author: Will Morris

#include "hpimage/hpimage.h"

// For now, only horizontal carving.
// end goal: specify target dimensions
int main(int argc, char* argv[]) {
    hpcarver::init();
}

// COMMON:
// -- Load image
// -- Write image
// -- Entry point (one terminal entry)
//    -- Delegate to CUDA library.
// -- Update image
//    -- Reduce size? Probably a common function.

// API:
// -- GetVertEnergy
//      -- Using dynamic programming, get energy in the vertical direction (up/down)
// -- GetHorizEnergy
//      -- Using dynamic programming, get energy in the horizontal direction (up/down)
// -- GetVertSeam
//      -- Using dynamic programming, get the minimum vertical seam.
// -- GetHorizSeam
//      -- Using dynamic programming, get the minimum horizontal seam.
// -- UpdateImage
//      -- Create a new image buffer not using the minimum seam
//      -- Possibly distinguish horizontal, vertical
