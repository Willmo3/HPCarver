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

    // Load an image using our defaults.
    // Heap allocating -- this could be large!
    Image *load_image(const char *filename) {
        auto *image = new Image(filename);
        image->type(TrueColorType);
        // Internal to hpcarver
        image->modifyImage();

        return image;
    }

    // Write an image to specified directory
    void write_image(Image *img, std::string &path) {
        img->write(path);
    }
}
