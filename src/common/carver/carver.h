//
// Created by morri2wj on 3/30/24.
//

#ifndef HPCARVER_CARVER_H
#define HPCARVER_CARVER_H

#include "../../../HPImage/hpimage.h"
#include "energy.h"
#include <vector>
#include <iostream>

namespace carver {

/**
 * Represents the API for a single carver class.
 * Why am I using a class?
 * All of these operations require two fields: an image and an energy matrix.
 * Additionally, many of the preparatory operations of these fields are inherently serial.
 *
 * Therefore, I am wrapping parts of the class with a standard serial behavior
 * And allowing the various implementations to define behaviors that may be parallelized
 *
 * Author: Will Morris
 */
class Carver {

// To prevent repeated reallocation, keeping image, energy fields.
private:
    hpimage::Hpimage *image;
    carver::Energy *energy;

    /**
     * An image must be at least 3x3. If not, complain.
     * @param hpimage image to perform assertions on
     */
    void assert_valid_dims() const;

    /**
     * Traverses the carver's energy matrix from left to right, updating energy based on base_energy + predecessors.
     * <b>Implemented by libraries!</b>
     */
    void horiz_energy();

    /**
     * Compute the minimum energy of each pixel in the vertical direction, storing in carver's energy memo structure.
     */
    void vert_energy();

// NOTE: ALL PUBLIC METHODS MUST BE IMPLEMENTED BY VARIOUS LIBRARY IMPLEMENTATIONS!
public:
    // ***** CONSTRUCTORS ***** //

    /**
     * HPCarver default constructor.
     * Initialize superclass fields to NULL.
     *
     */
    Carver();

    /**
     * HPCarver constructor.
     *
     * @param image Image to operate on.
     */
    Carver(hpimage::Hpimage *image, Energy *energy);


    // ***** ACCESSORS ***** //
    virtual hpimage::Hpimage *get_image();

    virtual carver::Energy *get_energy();

    // ***** MISC HELPER METHODS ***** //

    /**
     * Resize this carver's image.
     * Repeatedly recalculates the energy matrix and removes the minimum energy seam.
     *
     * @param new_width Width of image after resizing
     * @param new_height Height of image after resizing
     */
    void resize(uint32_t new_width, uint32_t new_height);

    /**
     * Get the base energy of a single pixel.
     * This is calculated using an energy gradient approach
     * considering the differences of adjacent colors
     *
     * @param row Row of pixel whose energy we're calculating
     * @param col Column of pixel whose energy we're calculating
     * @return the energy
     */
    uint32_t pixel_energy(uint32_t col, uint32_t row) const;

    // ***** IMPLEMENTED BY VARIOUS LIBRARIES *****//

    // These functions require the most computation
    // And offer the most opportunities for parallelization.

    /**
     * Given an end index and a carver, traverse the carver's energy matrix
     * Finding the minimum connected index at each point.
     * <b>Implemented by libraries!</b>
     *
     * @return The minimum seam, in the correct direction.
     */
    std::vector<uint32_t> min_vert_seam();

    /**
     * Given an end index and a carver, traverse the carver's energy matrix
     * Finding the minimum connected index at each point.
     * <b>Implemented by libraries!</b>
     *
     * @param carver carver to use.
     * @return The minimum seam, in the correct direction.
     */
    std::vector<uint32_t> min_horiz_seam();

    /**
     * Remove a horizontal seam from the image.
     * Updates the given image object.
     *
     * @param seam Seam to remove.
     */
    void remove_horiz_seam(std::vector<uint32_t> &seam);

    /**
     * Remove a vertical seam from the image.
     * Updates the given image object.
     *
     * @param seam Seam to remove.
     */
    void remove_vert_seam(std::vector<uint32_t> &seam);
};
} // namespace carver

#endif //HPCARVER_CARVER_H
