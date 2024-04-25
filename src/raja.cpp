/**
 * Raja implementation of HPCarver. This implements API-defined functions using the RAJA performance portability suite.
 *
 * Author: Will Morris
 */
#include <cassert>
#include <algorithm>

#include "carver.h"
#include "energy.h"
#include "RAJA/RAJA.hpp"

namespace carver {
// Starting out with just sequential execution
using policy = RAJA::seq_exec;

// Carver constructor
Carver::Carver(hpimage::Hpimage &image):
    image(image), energy(Energy(image.cols(), image.rows()))
    {
        assert_valid_dims();
    }


// ***** HORIZONTAL SEAM CALCULATORS ***** //

void Carver::horiz_energy() {
    // Horizontal traversal will go over all rows in each column.
    RAJA::RangeSegment range(0, energy.rows());
    // Prime first column with base energies
    RAJA::forall<policy>(range, [=] (int row) {
        energy.set_energy(0, row, pixel_energy(0, row));
    });

    // Now set energy to minimum of three neighbors.
    for (auto col = 1; col < energy.cols(); ++col) {
        RAJA::forall<policy>(range, [=] (int row) {
            // No wrapping
            auto neighbor_energies = energy.get_left_predecessors(col, row);

            // Energy = local energy + min(neighbors)
            uint32_t local_energy = pixel_energy(col, row);
            local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
            energy.set_energy(col, row, local_energy);
        });
    }
}

std::vector<uint32_t> Carver::min_horiz_seam() {
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


// ***** VERTICAL SEAM CALCULATORS ***** //

void Carver::vert_energy() {
    // Vertical seam direction: top to bottom
    RAJA::RangeSegment range(0, energy.cols());
    // Prime first row with base energies
    RAJA::forall<policy>(range, [=] (int col) {
        energy.set_energy(col, 0, pixel_energy(col, 0));
    });

    // This is one of the larger opportunities for parallelism.
    // Set energy to minimum of three above neighbors.
    for (auto row = 1; row < energy.rows(); ++row) {
        RAJA::forall<policy>(range, [=] (int col) {
            // Note: no wrapping in seams!
            auto neighbor_energies = energy.get_top_predecessors(col, row);

            // energy = local energy + min(neighbors)
            uint32_t local_energy = pixel_energy(col, row);
            local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
            energy.set_energy(col, row, local_energy);
        });
    }
}

std::vector<uint32_t> Carver::min_vert_seam() {
    uint32_t bottom_row = energy.rows() - 1;
    auto seam = std::vector<uint32_t>{};

    // Default: row 0 of the last column contains the minimum energy.
    // Invariant: there will be at least two rows to consider.
    uint32_t min_col = 0;
    uint32_t min_energy = energy.get_energy(0, bottom_row);

    for (auto col = 1; col < energy.cols(); ++col) {
        uint32_t current_energy = energy.get_energy(col, bottom_row);
        if (current_energy < min_energy) {
            min_col = col;
            min_energy = current_energy;
        }
    }

    seam.push_back(min_col);

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


// ***** SEAM REMOVERS ***** //

void Carver::remove_horiz_seam(std::vector<uint32_t> &seam) {
    // Must be exactly one row to remove from each column.
    assert(seam.size() == image.cols());
    RAJA::RangeSegment range(0, image.cols());

    // Prime first row with base energies
    RAJA::forall<policy>(range, [=] (int col) {
        auto index = seam[col];
        assert(index >= 0 && index < image.rows());

        // Shift all pixels below this up one.
        for (auto row = index; row < image.rows() - 1; ++row) {
            hpimage::pixel below = image.get_pixel(col, row + 1);
            image.set_pixel(col, row, below);
        }
    });

    // Finally, cut the last row from the pixel.
    energy.cut_row();
    image.cut_row();
}

void Carver::remove_vert_seam(std::vector<uint32_t> &seam) {
    // Must be exactly one column to remove from each row.
    assert(seam.size() == image.rows());
    RAJA::RangeSegment range(0, image.rows());

    // Shift every pixel after a given image over.
    RAJA::forall<policy>(range, [=] (int row) {
        auto index = seam[row];
        assert(index >= 0 && index < image.cols());

        // Shift all pixels after this one back
        for (auto col = index; col < image.cols() - 1; ++col) {
            hpimage::pixel next = image.get_pixel(col + 1, row);
            image.set_pixel(col, row, next);
        }
    });

    // Finally, with all pixels shifted over, time to trim the image!
    energy.cut_col();
    image.cut_col();
}
} // namespace carver
