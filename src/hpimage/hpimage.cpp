// Common functionality related to loading, writing images.
// These will be relied upon by various hpcarver renditions.
// Author: Will Morris

#include "hpimage.h"
using namespace Magick;

namespace hpcarver {
    void init() {
        // We don't use custom args here, do not provide Magick with args.
        InitializeMagick(nullptr);
    }
}
