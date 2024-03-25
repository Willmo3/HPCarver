// Entry file for HPcarver
// Author: Will Morris

#include <Magick++.h>
#include <iostream>

using namespace Magick;

int main(int argc, char* argv[]) {
    InitializeMagick(*argv);
    std::cout << "Hello, world!" << std::endl;
}
