#include "../../common/carver/carver.h"
#include "cuda_energy.h"
#include "../image/cuda_image.h"

#include <cassert>
#include <algorithm>


// **** INDEXING HELPERS: ***** //

// Wrap an index based on length.
__device__ uint32_t wrap_index(int64_t index, size_t length) {
    return (index + length) % length;
}

// Get a pixel from a CudaImage buffer
__device__ hpimage::pixel get_pixel(hpc_cuda::CudaImageStruct image, uint32_t col, uint32_t row) {
    assert(col < image.current_cols);
    assert(row < image.current_rows);

    // Stride: skip through all the columns of previous rows.
    return image.pixels[row * image.base_cols + col];
}

// Get an energy value from a CudaImage buffer
__device__ uint32_t get_energy(hpc_cuda::CudaEnergyStruct energy, uint32_t col, uint32_t row) {
    assert(col < energy.current_cols);
    assert(row < energy.current_rows);

    // Stride: skip through all the columns of previous rows.
    return energy.energy[row * energy.base_cols + col];
}

// Set an energy value to new_value
__device__ void set_energy(hpc_cuda::CudaEnergyStruct energy, uint32_t new_value, uint32_t col, uint32_t row) {
    assert(col < energy.current_cols);
    assert(row < energy.current_rows);

    energy.energy[row * energy.base_cols + col] = new_value;
}

// Set an energy value to new_value
__device__ void set_pixel(hpc_cuda::CudaImageStruct *image, hpimage::pixel new_value, uint32_t col, uint32_t row) {
    assert(col < image->current_cols);
    assert(row < image->current_rows);

    image->pixels[row * image->base_cols + col] = new_value;
}


// ***** ENERGY HELPERS ***** //

/**
 * Get the gradient energy difference between two pixels.
 *
 * @param p1 First pixel to consider.
 * @param p2 Second pixel to consider.
 * @param retval Pointer to integer to place energy in.
 */
__device__ uint32_t gradient_energy(hpimage::pixel p1, hpimage::pixel p2) {
    auto energy = 0;

    auto red_diff = p1.red - p2.red;
    auto green_diff = p1.green - p2.green;
    auto blue_diff = p1.blue - p2.blue;

    // Square differences w/o importing pow fn
    energy += red_diff * red_diff;
    energy += green_diff * green_diff;
    energy += blue_diff * blue_diff;

    return energy;
}

/**
 * Get the basic energy of a pixel at col, row.
 * Does not consider previous neighbor energy in memo structure.
 *
 * @param c_energy Cuda energy struct containing energy matrix.
 * @param col Col to consider.
 * @param row Row to consider.
 */
__device__ uint32_t pixel_energy(hpc_cuda::CudaImageStruct c_image, uint32_t col, uint32_t row) {
    assert(col < c_image.current_cols);
    assert(row < c_image.current_rows);

    // Casting here to allow the standard uint32_t API for external calls
    // While avoiding underflow with internal ones.
    auto signed_col = (int64_t) col;
    auto signed_row = (int64_t) row;

    uint32_t left_col = wrap_index(signed_col - 1, c_image.current_cols);
    uint32_t right_col = wrap_index(signed_col + 1, c_image.current_cols);
    uint32_t top_row = wrap_index(signed_row + 1, c_image.current_rows);
    uint32_t bottom_row = wrap_index(signed_row - 1, c_image.current_rows);

    hpimage::pixel left = get_pixel(c_image, left_col, row);
    hpimage::pixel right = get_pixel(c_image, right_col, row);
    hpimage::pixel top = get_pixel(c_image, col, top_row);
    hpimage::pixel bottom = get_pixel(c_image, col, bottom_row);

    return gradient_energy(left, right) + gradient_energy(top, bottom);
}

/**
 * Get the minimum energy of the energy struct's left neighbors.
 * There will be 2-3 neighbors, depending on where in the image the pixel is.
 *
 * @param c_energy Energy struct to grab energies from.
 * @param col Column to get energy from.
 * @param row Row to get energy from.
 * @return The minimum energy of the neighbors.
 */
__device__ uint32_t min_left_energy(hpc_cuda::CudaEnergyStruct c_energy, uint32_t col, uint32_t row) {
    assert(col > 0);
    uint32_t left_col = col - 1;

    // While we allow wrapping for calculating basic energies, there is no wrapping in seams.
    // Therefore, each pixel is allowed only to consider the neighbors they have.
    int64_t min_energy = -1;
    if (row > 0) {
        uint32_t top_energy = get_energy(c_energy, left_col, row - 1);
        min_energy = top_energy;
    }

    uint32_t middle_energy = get_energy(c_energy, left_col, row);
    if (min_energy == -1 || middle_energy < min_energy) {
        min_energy = middle_energy;
    }

    if (row + 1 < c_energy.current_rows) {
        uint32_t bottom_energy = get_energy(c_energy, left_col, row + 1);
        if (bottom_energy < min_energy) {
            min_energy = bottom_energy;
        }
    }

    return min_energy;
}

/**
 * Get the minimum energy of a cells top neighbors.
 *
 * @param c_energy Energy matrix to consider.
 * @param col column of cell
 * @param row row of cell
 * @return The minimum energy of the top neighbors.
 */
__device__ uint32_t min_top_energy(hpc_cuda::CudaEnergyStruct c_energy, uint32_t col, uint32_t row) {
    assert(row > 0);
    uint32_t upper_row = row - 1;

    int64_t min_energy = -1;
    if (col > 0) {
        uint32_t left_energy = get_energy(c_energy, col - 1, upper_row);
        min_energy = left_energy;
    }

    uint32_t middle_energy = get_energy(c_energy, col, upper_row);
    if (min_energy == -1 || middle_energy < min_energy) {
        min_energy = middle_energy;
    }

    if (col + 1 < c_energy.current_cols) {
        uint32_t right_energy = get_energy(c_energy, col + 1, upper_row);
        if (right_energy < min_energy) {
            min_energy = right_energy;
        }
    }

    return min_energy;
}


// ***** ENERGY NEIGHBOR CALCULATORS ***** //

/**
 * Given an energy matrix, compute the minimum energy of col considering previous neighbor's energies.
 *
 * @param energy Energy matrix to use.
 * @param col Column to start from. Must be greater than zero, because we're considering backwards neighbor energies.
 */
__global__ void horiz_energy_neighbor(hpc_cuda::CudaEnergyStruct c_energy,
                                      hpc_cuda::CudaImageStruct c_image, uint32_t col) {
    assert(col > 0 && col < c_energy.current_cols);
    assert(c_energy.current_rows > 0);

    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int row = start; row < c_energy.current_rows; row += stride) {
        uint32_t local_energy = pixel_energy(c_image, col, row) + min_left_energy(c_energy, col, row);
        set_energy(c_energy, local_energy, col, row);
    }
}

/**
 * Given an energy matrix, compute the minimum energy of row considering preceeding row energies.
 * @param energy Energy matrix to use.
 * @param row Row to start from. Must be greater than zero -- considering predecessor energy.
 */
__global__ void vert_energy_neighbor(hpc_cuda::CudaEnergyStruct c_energy,
                                     hpc_cuda::CudaImageStruct c_image, uint32_t row) {
    assert(row > 0 && row < c_energy.current_rows);
    assert(c_energy.current_rows > 0);

    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int col = start; col < c_energy.current_cols; col += stride) {
        uint32_t local_energy = pixel_energy(c_image, col, row) + min_top_energy(c_energy, col, row);
        set_energy(c_energy, local_energy, col, row);
    }
}


// ***** SEAM REMOVERS ***** //

// These could be implemented in CUDA, but this computation does not appear to be the performance bottleneck
// Therefore, to save time, I've decided against writing CUDA versions at this point.

/**
 * Remove a vertical seam from c_image.
 * ASSUMPTION: each col in seam is adjacent (perhaps should be improved).
 *
 * @param c_image image to remove seam from.
 * @param seam Seam to remove
 * @param seam_len Length of seam to remove; should be equal to c_image->rows.
 *
 */
__global__ void remove_vert_seam(hpc_cuda::CudaImageStruct *c_image, uint32_t *seam, size_t seam_len) {
    assert(seam_len == c_image->current_rows);
}

namespace carver {

// ***** HORIZONTAL SEAM CALCULATORS ***** //

void Carver::horiz_energy() {
    for (auto row = 0; row < energy->rows(); ++row) {
        energy->set_energy(0, row, pixel_energy(0, row));
    }

    // Now set energy to minimum of three neighbors.
    for (auto col = 1; col < energy->cols(); ++col) {
        horiz_energy_neighbor<<<10, 1024>>>(((hpc_cuda::CudaEnergy *) energy)->to_struct(),
                                            ((hpc_cuda::CudaImage *) image)->to_struct(), col);
        cudaDeviceSynchronize();
    }
}

std::vector<uint32_t> Carver::min_horiz_seam() {
    // Now, prime the reverse traversal with the minimum above energy->
    uint32_t back_col = energy->cols() - 1;
    auto seam = std::vector<uint32_t>{};

    // Default: row 0 of the last column contains the minimum energy->
    // Invariant: there will be at least two rows to consider.
    uint32_t min_row = 0;
    uint32_t min_energy = energy->get_energy(back_col, 0);

    for (auto row = 1; row < energy->rows(); ++row) {
        uint32_t current_energy = energy->get_energy(back_col, row);
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
        min_energy = energy->get_energy(col, min_row);
        // Check if the upper or lower neighbors are actually better choices.
        if (row > 0 && min_energy > energy->get_energy(col, row - 1)) {
            min_row = row - 1;
            min_energy = energy->get_energy(col, row - 1);
        }
        if (row + 1 < energy->rows() && min_energy > energy->get_energy(col, row + 1)) {
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
    // Prime memo structure with base energies of first pixel row.
    for (auto col = 0; col < energy->cols(); ++col) {
        energy->set_energy(col, 0, pixel_energy(col, 0));
    }

    // Now, calculate energies, considering neighbors.
    for (auto row = 1; row < energy->rows(); ++row) {
        vert_energy_neighbor<<<10, 1024>>>(((hpc_cuda::CudaEnergy *) energy)->to_struct(),
                                            ((hpc_cuda::CudaImage *) image)->to_struct(), row);
        cudaDeviceSynchronize();
    }
}

std::vector<uint32_t> Carver::min_vert_seam() {
    uint32_t bottom_row = energy->rows() - 1;
    auto seam = std::vector<uint32_t>{};

    // Default: row 0 of the last column contains the minimum energy->
    // Invariant: there will be at least two rows to consider.
    uint32_t min_col = 0;
    uint32_t min_energy = energy->get_energy(0, bottom_row);

    for (auto col = 1; col < energy->cols(); ++col) {
        uint32_t current_energy = energy->get_energy(col, bottom_row);
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
        min_energy = energy->get_energy(min_col, row);
        // Check if the upper or lower neighbors are actually better choices.
        if (col > 0 && min_energy > energy->get_energy(col - 1, row)) {
            min_col = col - 1;
            min_energy = energy->get_energy(col - 1, row);
        }
        if (col + 1 < energy->cols() && min_energy > energy->get_energy(col + 1, row)) {
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
    assert(seam.size() == image->cols());

    for (auto col = 0; col < image->cols(); ++col) {
        auto index = seam[col];
        assert(index < image->rows());

        // Shift all pixels below this up one.
        for (auto row = index; row < image->rows() - 1; ++row) {
            hpimage::pixel below = image->get_pixel(col, row + 1);
            image->set_pixel(col, row, below);
        }
    }
    // Finally, cut the last row from the pixel.
    energy->cut_row();
    image->cut_row();
}

void Carver::remove_vert_seam(std::vector<uint32_t> &seam) {
    // Must be exactly one column to remove from each row.
    assert(seam.size() == image->rows());

    // Shift every pixel after a given image over.
    // Then reduce image size by one.
    for (auto row = 0; row < image->rows(); ++row) {
        auto index = seam[row];
        assert(index < image->cols());

        // Shift all pixels after this one back
        for (auto col = index; col < image->cols() - 1; ++col) {
            hpimage::pixel next = image->get_pixel(col + 1, row);
            image->set_pixel(col, row, next);
        }
    }
    // Finally, with all pixels shifted over, time to trim the image!
    energy->cut_col();
    image->cut_col();
}
} // namespace carver
