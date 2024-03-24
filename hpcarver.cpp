// Entry file for HPcarver
// Author: Will Morris
#include <iostream>
#include "netpbm.h"

void usage();

int main(int argc, char* argv[]) {
    if (argc != 2) {
        usage();
        return 1;
    }
    std::cout << "Hello, world!" << std::endl;
}

void usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "hpcarver [source.ppm]" << std::endl;
}
