/**
 * OpenMP implementation of HPCarver. This implements API-defined functions using OpenMP manner.
 *
 * Author: Will Morris
 */
#include <cassert>
#include <algorithm>

#include "carver.h"
#include "energy.h"

namespace carver {

Carver prepare_carver(hpimage::Hpimage &image) {
    // In our case, it's enough to just return a standard carver.
    return Carver(image);
}

// Carver constructor
Carver::Carver(hpimage::Hpimage &image):
        image(image), energy(Energy(image.cols(), image.rows()))
{ assert_valid_dims(); }

std::vector<uint32_t> Carver::horiz_seam() {
    assert_valid_dims();

    // At the end, pick any of the pixels with minimal memoized energy. Then, find the lowest energy
    // Adjacent pixel in the next row. Repeat until the end, adding to the seam vector.

    // Return the reversed vector.

    // Generate energy matrix
    // Horizontal seam direction: left to right.
    // Prime memo structure with base energies of first pixel column.

    // OPENMP: on large images, should see benefit even in first row.
#   pragma omp parallel for default(none) shared(energy)
    for (auto row = 0; row < energy.rows(); ++row) {
        energy.set_energy(0, row, pixel_energy(0, row));
    }

    // Now set energy to minimum of three neighbors.

    // OMP: Loop carried dependency -- previous column must have been initialized
    for (auto col = 1; col < energy.cols(); ++col) {

        // OMP: col shared bc all parallelization occurs within a given col.
        // Energy: only reads from behind and writes once to each position in current col
#       pragma omp parallel for default(none) shared(col, energy)
        for (auto row = 0; row < energy.rows(); ++row) {
            // No wrapping
            auto neighbor_energies = energy.get_left_predecessors(col, row);

            // Energy = local energy + min(neighbors)
            uint32_t local_energy = pixel_energy(col, row);
            local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
            energy.set_energy(col, row, local_energy);
        }
    }

    // Now, prime the reverse traversal with the minimum above energy.
    uint32_t back_col = energy.cols() - 1;
    auto seam = std::vector<uint32_t>{};

    // Default: row 0 of the last column contains the minimum energy.
    // Invariant: there will be at least two rows to consider.
    uint32_t min_row = 0;
    uint32_t min_energy = energy.get_energy(back_col, 0);

    for (auto row = 1; row < energy.rows(); ++row) {
        uint32_t current_energy = energy.get_energy(back_col, row);
        if (current_energy < min_energy) {
            min_row = row;
            min_energy = current_energy;
        }
    }
    seam.push_back(min_row);

    // Find the rest of the seam, using only the three predecessors of each node.
    // Using wider signed form to prevent underflow
    for (int64_t col = back_col - 1; col >= 0; --col) {
        // Get the previous index from which to grab neighbors.
        auto row = seam.back();
        min_row = row;
        min_energy = energy.get_energy(col, min_row);
        // Check if the upper or lower neighbors are actually better choices.
        if (row > 0 && min_energy > energy.get_energy(col, row - 1)) {
            min_row = row - 1;
            min_energy = energy.get_energy(col, row - 1);
        }
        if (row + 1 < energy.rows() && min_energy > energy.get_energy(col, row + 1)) {
            min_row = row + 1;
        }
        seam.push_back(min_row);
    }

    // Finally, reverse seam so that it goes in the natural rear-forward order.
    std::reverse(seam.begin(), seam.end());
    return seam;
}

std::vector<uint32_t> Carver::vertical_seam() {
    assert_valid_dims();

    // Vertical seam direction: top to bottom
    // Prime memo structure with base energies of first pixel row.

    // OMP: initialize energy
#   pragma omp parallel for default(none) shared(energy)
    for (auto col = 0; col < energy.cols(); ++col) {
        energy.set_energy(col, 0, pixel_energy(col, 0));
    }

    // This is one of the larger opportunities for parallelism.
    // Set energy to minimum of three above neighbors.

    for (auto row = 1; row < energy.rows(); ++row) {

        // OMP: row shared bc all parallelization occurs within a given row.
        // Energy: only reads from above and writes once to each position in current row
#       pragma omp parallel for default(none) shared(row, energy)
        for (auto col = 0; col < energy.cols(); ++col) {
            // Note: no wrapping in seams!
            auto neighbor_energies = energy.get_top_predecessors(col, row);

            // energy = local energy + min(neighbors)
            uint32_t local_energy = pixel_energy(col, row);
            local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
            energy.set_energy(col, row, local_energy);
        }
    }

    // Now, prime the reverse traversal with the minimum above energy.
    uint32_t bottom_row = energy.rows() - 1;
    auto seam = std::vector<uint32_t>{};

    // Default: row 0 of the last column contains the minimum energy.
    // Invariant: there will be at least two rows to consider.
    uint32_t min_col = 0;
    uint32_t min_energy = energy.get_energy(0, bottom_row);

    // OMP: mutating min_energy, not worth parallelizing -- would need to be basically synchronized.
    for (auto col = 1; col < energy.cols(); ++col) {
        uint32_t current_energy = energy.get_energy(col, bottom_row);
        if (current_energy < min_energy) {
            min_col = col;
            min_energy = current_energy;
        }
    }

    // Problem: tight coupling.
    // This should be generic enough to work for any amount

    seam.push_back(min_col);

    // OMP: state of seam is loop carried dependency, no opportunity for parallelization.

    // Find the rest of the seam, using only the three predecessors of each node.
    // Using wider signed form to prevent underflow
    for (int64_t row = bottom_row - 1; row >= 0; --row) {
        // Get the previous index from which to grab neighbors
        auto col = seam.back();
        min_col = col;
        min_energy = energy.get_energy(min_col, row);
        // Check if the upper or lower neighbors are actually better choices.
        if (col > 0 && min_energy > energy.get_energy(col - 1, row)) {
            min_col = col - 1;
            min_energy = energy.get_energy(col - 1, row);
        }
        if (col + 1 < energy.cols() && min_energy > energy.get_energy(col + 1, row)) {
            min_col = col + 1;
        }
        seam.push_back(min_col);
    }

    // Reverse the seam so traversal happens in expected direction.
    std::reverse(seam.begin(), seam.end());
    return seam;
}

uint32_t Carver::pixel_energy(int32_t col, int32_t row) {
    assert(col >= 0 && col < image.cols());
    assert(row >= 0 && row < image.rows());

    uint32_t left_col = wrap_index(col - 1, image.cols());
    uint32_t right_col = wrap_index(col + 1, image.cols());
    uint32_t top_row = wrap_index(row + 1, image.rows());
    uint32_t bottom_row = wrap_index(row - 1, image.rows());

    hpimage::pixel left = image.get_pixel(left_col, row);
    hpimage::pixel right = image.get_pixel(right_col, row);
    hpimage::pixel top = image.get_pixel(col, top_row);
    hpimage::pixel bottom = image.get_pixel(col, bottom_row);

    return gradient_energy(left, right) + gradient_energy(top, bottom);
}

void Carver::remove_horiz_seam(std::vector<uint32_t> &seam) {
    // Must be exactly one row to remove from each column.
    assert(seam.size() == image.cols());

    // OMP: each thread only accesses a single column of the image
    // And only reads from seam
#   pragma omp parallel for default(none) shared(image, seam)
    for (auto col = 0; col < image.cols(); ++col) {
        auto index = seam[col];
        // Due to issues with version of gcc on cluster, commenting out this asssertion.
        // Refer to: https://stackoverflow.com/questions/47081274/openmp-error-w-13-not-specified-in-enclosing-parallel
        // assert(index >= 0 && index < image.rows());

        // Shift all pixels below this up one.
        for (auto row = index; row < image.rows() - 1; ++row) {
            hpimage::pixel below = image.get_pixel(col, row + 1);
            image.set_pixel(col, row, below);
        }
    }
    // Finally, cut the last row from the image.
    energy.cut_row();
    image.cut_row();
}

void Carver::remove_vert_seam(std::vector<uint32_t> &seam) {
    // Must be exactly one column to remove from each row.
    assert(seam.size() == image.rows());

    // OMP each thread only accesses a single row of the image
    // And only reads from seam
#   pragma omp parallel for default(none) shared(image, seam)
    for (auto row = 0; row < image.rows(); ++row) {
        auto index = seam[row];
        // Due to issues with version of gcc on cluster, commenting out this asssertion.
        // Refer to: https://stackoverflow.com/questions/47081274/openmp-error-w-13-not-specified-in-enclosing-parallel
        // assert(index >= 0 && index < image.cols());

        // Shift all pixels after this one back
        for (auto col = index; col < image.cols() - 1; ++col) {
            hpimage::pixel next = image.get_pixel(col + 1, row);
            image.set_pixel(col, row, next);
        }
    }
    // Finally, with all pixels shifted over, time to trim the image!
    energy.cut_col();
    image.cut_col();
}
} // namespace carver
