//
// Created by morri2wj on 3/29/24.
//

#ifndef HPCARVER_HPIMAGE_H
#define HPCARVER_HPIMAGE_H

#include <Magick++.h>

namespace hpcarver {
    /**
     * Initialization function for hpcarver
     * ImageMagick needs the path to the program to properly link!
     */
    void init(const char *path);

    /**
     * Load an image from filesystem given filename
     * Use our defaults.
     * Heap allocate -- this could be big!
     * @param filename name of the file to use
     * @return the loaded image
     */
    Magick::Image *load_image(const char *filename);

    /**
     * @param image image to get pixels from
     * @return pixels from the image
     */
    Magick::Pixels *get_pixels(Magick::Image &image);
}

#endif //HPCARVER_HPIMAGE_H
