//
// Created by will on 4/7/24.
//

#ifndef HPCARVER_ENERGY_H
#define HPCARVER_ENERGY_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace carver {

/**
 * The energy class is a reusable, resizable matrix of energy values.
 *
 * @author Will Morris
 */
class Energy {
protected:
    size_t base_cols;
    size_t base_rows;
    size_t current_rows;
    size_t current_cols;
    uint32_t *energy;

    /**
     * Allocate memory for the energy buffer.
     * Virtual -- allow for extensibility with libraries like CUDA.
     * @param size Size of the memory region to allocate.
     * @return Pointer to allocated memory region.
     */
    virtual uint32_t *alloc(uint32_t size);

public:
    /**
     * Default energy constructor.
     * Allows for inheritance
     */
    Energy();

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

    // PREDECESSOR ACCESSORS
    // Useful for dynamic programming directional traversal.

    /**
     * Get the energies of the predecessors of a given pixel from above (i.e. its three top neighbors).
     * Useful for vertical traversal of data.
     *
     * @param col Column of pixel to consider
     * @param row Row of pixel to consider
     * @return A predecessor vector of at most size 3.
     */
    std::vector<uint32_t> get_top_predecessors(uint32_t col, uint32_t row) const;

    /**
     * Get the energies of the predecessors of a given pixel from the left (i.e. its three left neighbors)
     * Useful for horizontal traversal of data.
     *
     * @param col Column of pixel to consider
     * @param row Row of pixel to consider
     * @return A predecessor vector of at most size 3.
     */
    std::vector<uint32_t> get_left_predecessors(uint32_t col, uint32_t row) const;

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

} // namespace carver

#endif //HPCARVER_ENERGY_H
