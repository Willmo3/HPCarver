// Common functionality related to loading, writing images.
// These will be relied upon by various hpcarver renditions.
// Author: Will Morris

#include <cassert>
#include "hpimage.h"
using namespace Magick;

namespace hpcarver {
void init() {
    InitializeMagick(nullptr);
}

// Load an image using our defaults.
// Heap allocating -- this could be large!
Image *load_image(const char *filename) {
    auto *image = new Image(filename);
    // Required by Magick++
    image->type(TrueColorType);
    image->modifyImage();
    return image;
}

// Write an image to specified directory
void write_image(Image *img, std::string &path) {
    img->write(path);
}

// Reduce width of image by one.
void cut_width(Magick::Image *img) {
    // Refuse to cut width down to 0!
    assert(img->columns() > 1);

    Geometry crop = Geometry(img->columns() - 1, img->rows());
    img->crop(crop);
}

// Reduce height of image by one.
void cut_height(Magick::Image *img) {
    // Refuse to cut height down to 0!
    assert(img->rows() > 1);

    Geometry crop = Geometry(img->columns(), img->rows() - 1);
    img->crop(crop);
} // End hpcarver namespace
}
