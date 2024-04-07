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
} // carver