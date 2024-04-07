//
// Created by will on 4/7/24.
//

#ifndef HPCARVER_ENERGY_H
#define HPCARVER_ENERGY_H

#include <cstddef>
#include <cstdint>

namespace carver {

/**
 * The energy class is a reusable, resizable matrix of energy values.
 *
 * @author Will Morris
 */
class Energy {
private:
    size_t base_cols;
    size_t base_rows;
    size_t current_rows;
    size_t current_cols;
    uint32_t *energy;

public:
    /**
     * Energy constructor.
     * This initializes a block of memory for an energy matrix.
     * @param cols Number of columns to initialize energy block with
     * @param rows Number of rows to initialize energy block with
     */
    Energy(uint32_t cols, uint32_t rows);

    /**
     * Energy destructor.
     * Will free heap-allocated resources (i.e. the energy block).
     */
    ~Energy();

    // ACCCESSORS

    uint32_t get_energy(uint32_t col, uint32_t row) const;

    // Get number of rows and columns
    size_t cols() const;

    size_t rows() const;

    // MUTATORS

    void set_energy(uint32_t col, uint32_t row, uint32_t new_energy);

    // These reduce width, height by one for SeamCarving
    // Note: these will fail if their relevant fields would be reduced to zero.
    void cut_row();

    void cut_col();
};

} // carver

#endif //HPCARVER_ENERGY_H
