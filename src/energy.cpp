//
// Created by will on 4/7/24.
//

#include <cassert>
#include <cstdlib>
#include "energy.h"

namespace carver {
    Energy::Energy(uint32_t cols, uint32_t rows) {
        assert(cols > 0 && rows > 0);
        base_cols = cols;
        current_cols = cols;
        base_rows = rows;
        current_rows = rows;

        energy = static_cast<uint32_t *>(calloc(rows * cols, sizeof(uint32_t)));
    }

    /**
     * Free heap-allocated energy vector
     */
    Energy::~Energy() {
        free(energy);
        energy = nullptr;
    }

    // ACCESSORS

    size_t Energy::cols() const {
        return current_cols;
    }

    size_t Energy::rows() const {
        return current_rows;
    }

    uint32_t Energy::get_energy(uint32_t col, uint32_t row) const {
        assert(col >= 0 && col < cols());
        assert(row >= 0 && row < rows());

        // Stride: skip through all the columns of previous rows.
        return energy[row * base_cols + col];
    }

    std::vector<uint32_t> Energy::get_top_predecessors(uint32_t col, uint32_t row) const {
        // For now, this will only be called on indexes with a left predecessor
        assert(row > 0);
        uint32_t upper_row = row - 1;

        // While we allow wrapping for calculating basic energies, there is no wrapping in seams.
        // Therefore, each pixel is allowed only to consider the neighbors they have.
        auto neighbor_energies = std::vector<uint32_t>();

        if (col > 0) {
            uint32_t left_energy = get_energy(col - 1, upper_row);
            neighbor_energies.push_back(left_energy);
        }

        uint32_t middle_energy = get_energy(col, upper_row);
        neighbor_energies.push_back(middle_energy);

        if (col + 1 < cols()) {
            uint32_t right_energy = get_energy(col + 1, upper_row);
            neighbor_energies.push_back(right_energy);
        }

        return neighbor_energies;
    }

    std::vector<uint32_t> Energy::get_left_predecessors(uint32_t col, uint32_t row) const {
        // For now, this will only be called on indexes with a left predecessor.
        assert(col > 0);
        uint32_t left_col = col - 1;

        // While we allow wrapping for calculating basic energies, there is no wrapping in seams.
        // Therefore, each pixel is allowed only to consider the neighbors they have.
        auto neighbor_energies = std::vector<uint32_t>();

        if (row > 0) {
            uint32_t top_energy = get_energy(left_col, row - 1);
            neighbor_energies.push_back(top_energy);
        }

        uint32_t middle_energy = get_energy(left_col, row);
        neighbor_energies.push_back(middle_energy);

        if (row + 1 < rows()) {
            uint32_t bottom_energy = get_energy(left_col, row + 1);
            neighbor_energies.push_back(bottom_energy);
        }

        return neighbor_energies;
    }

    // MUTATORS

    void Energy::set_energy(uint32_t col, uint32_t row, uint32_t new_energy) {
        assert(col >= 0 && col < cols());
        assert(row >= 0 && row < rows());
        energy[row * base_cols + col] = new_energy;
    }

    void Energy::cut_col() {
        // Program invariant: height must be greater than one if cut.
        assert(current_cols > 1);
        current_cols -= 1;
    }

    void Energy::cut_row() {
        // Program invariant: width must be greater than one if cut.
        assert(current_rows > 1);
        current_rows -= 1;
    }
} // namespace carver