//
// Created by morri2wj on 3/29/24.
//

#ifndef HPCARVER_HPIMAGE_H
#define HPCARVER_HPIMAGE_H

#include <Magick++.h>

namespace hpcarver {
/**
 * Initialization function for hpcarver
 */
void init();

/**
 * Load an image from filesystem given filename
 * Use our defaults.
 * Heap allocate -- this could be big!
 * @param filename name of the file to use
 * @return the loaded image
 */
Magick::Image *load_image(const char *filename);

/**
 * Write an image to file
 * @param img image to write
 * @param path path to file to write
 */
void write_image(Magick::Image *img, std::string &path);

/**
 * Crop from the edges of an image.
 * new_width should be less than img->width!
 *
 * @param img Image to resize
 */
void cut_width(Magick::Image *img);

/**
 * Crop from the top and bottom of an image.
 * new_height should be less than img->height!
 *
 * @param img Image to resize
 */
void cut_height(Magick::Image *img);
} // End hpcarver namespace

#endif //HPCARVER_HPIMAGE_H
