// Entry file for HPcarver
// Author: Will Morris

#include "hpimage/hpimage.h"

// For now, only horizontal carving.
// end goal: specify target dimensions
int main(int argc, char* argv[]) {
    hpcarver::init(*argv);
}
