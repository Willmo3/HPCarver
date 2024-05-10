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
    void assert_valid_dims();

    /**
     * Calculate a wrapped index over a dimension.
     * The formula is (index + length) % length
     * So that indexes of length will wrap to 0
     * And indexes of -1 will wrap to length -1.
     *
     * @param index Base index to wrap. Int64_t sizing prevents underflow.
     * @param length Length of dimension
     * @return The wrapped index
     */
    static uint32_t wrap_index(int64_t index, uint32_t length);

    /**
     * Calculate the gradient energy of two pixels.
     * This is simply the energy difference of their corresponding RGB color fields
     * Squared to ensure positivity and penalize small differences.
     *
     * @param p1 First pixel to consider.
     * @param p2 Second pixel to consider.
     * @return The gradient energy.
     */
    static uint32_t gradient_energy(hpimage::pixel p1, hpimage::pixel p2);

    /**
     * Traverses the carver's energy matrix from left to right, updating energy based on base_energy + predecessors.
     * <b>Implemented by libraries!</b>
     */
    void horiz_energy();

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
     * Compute the minimum energy of each pixel in the vertical direction, storing in carver's energy memo structure.
     */
    void vert_energy();

    /**
     * Given an end index and a carver, traverse the carver's energy matrix
     * Finding the minimum connected index at each point.
     * <b>Implemented by libraries!</b>
     *
     * @return The minimum seam, in the correct direction.
     */
    std::vector<uint32_t> min_vert_seam();

// NOTE: ALL PUBLIC METHODS MUST BE IMPLEMENTED BY VARIOUS LIBRARY IMPLEMENTATIONS!
public:
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

    /**
     * Return the minimum energy horizontal seam for this image
     * @return Minimum energy horizontal seam
     */
    std::vector<uint32_t> horiz_seam();

    /**
     * Given an image, return the minimum energy vertical seam
     * @return Minimum energy vertical seam
     */
    std::vector<uint32_t> vertical_seam();

    /**
     * Get the base energy of a single pixel.
     * This is calculated using an energy gradient approach
     * considering the differences of adjacent colors
     *
     * @param row Row of pixel whose energy we're calculating
     * @param col Column of pixel whose energy we're calculating
     * @return the energy
     */
    uint32_t pixel_energy(uint32_t col, uint32_t row);

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

    /**
     * ACCESSORS
     */
     virtual hpimage::Hpimage *get_image();
     virtual carver::Energy *get_energy();

};
} // namespace carver

#endif //HPCARVER_CARVER_H
