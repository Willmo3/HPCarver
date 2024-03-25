// Entry file for HPcarver
// Author: Will Morris

// BE ADVISED: your IDE probably won't find this!
// This is because ImageMagick has to use a helper script to actually compile; see the makefile.
// If you try to make it, you'll see it works. :/
#include <Magick++.h>
#include <iostream>

void usage();

using namespace Magick;

int main(int argc, char* argv[]) {
    InitializeMagick(*argv);
    std::cout << "Hello, world!" << std::endl;
}
