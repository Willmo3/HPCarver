// Entry file for HPcarver
// Author: Will Morris

#include <Magick++.h>
#include <iostream>

using namespace Magick;

// For now, only horizontal carving.
// end goal: specify target dimensions
int main(int argc, char* argv[]) {
    InitializeMagick(*argv);
    std::cout << "Hello, world!" << std::endl;
}

// What to do?
// Driver for c
