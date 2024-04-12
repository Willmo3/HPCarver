/**
 * Common files for all implementations of HPCarver.
 * These are assumed to all run serially.
 *
 * Author: Will Morris
 */

#include "carver.h"

using namespace carver;

void Carver::assert_valid_dims() {
    if (image.cols() < 3) {
        std::cerr <<
                  "ERROR: Carving: minimum image width three"
                  << std::endl;
        exit(EXIT_FAILURE);
    } else if (image.rows() < 2) {
        std::cerr <<
                  "ERROR: Carving: minimum image height three"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

uint32_t Carver::wrap_index(int32_t index, uint32_t length) {
    // Any negative value wraps around the other side.
    return (index + length) % length;
}

uint32_t Carver::gradient_energy(hpimage::pixel p1, hpimage::pixel p2) {
    auto energy = 0;

    auto red_diff = p1.red - p2.red;
    auto green_diff= p1.green - p2.green;
    auto blue_diff = p1.blue - p2.blue;

    // Square differences w/o importing pow fn
    energy += red_diff * red_diff;
    energy += green_diff * green_diff;
    energy += blue_diff * blue_diff;

    return energy;
}


