#include "../carver.h"
#include "cuda_energy.h"
#include "cuda_image.h"

#include <cassert>
#include <algorithm>

// Print usage data
void usage();

// Indexes for argument parsing
#define ARG_COUNT 5
#define SOURCE_INDEX 1
#define TARGET_INDEX 2
#define WIDTH_INDEX 3
#define HEIGHT_INDEX 4

// ***** PROGRAM ENTRY POINT ***** //

// if we're using CUDA, we want main to create cuda energy and cuda image
int main(int argc, char *argv[]) {
    // For now, default to sequential execution

    // There should be 5 options, including executable name
    if (argc != ARG_COUNT) {
        usage();
        exit(EXIT_FAILURE);
    }

    char *source_path = argv[SOURCE_INDEX];
    char *out_path = argv[TARGET_INDEX];
    uint32_t new_width = std::stoi(argv[WIDTH_INDEX]);
    uint32_t new_height = std::stoi(argv[HEIGHT_INDEX]);

    if (new_width < 3) {
        std::cerr << "ERROR: Must specify new image width greater than three!" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (new_height < 3) {
        std::cerr << "ERROR: Must specify new image height greater than three!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Load image
    auto image = hpc_cuda::CudaImage(source_path);

    // Ensure that image is being shrunk -- for now, this is all that's supported!
    if (new_width > image.cols() || new_height > image.rows()) {
        std::cerr << "ERROR: HPCarver supports shrinking. New image dimensions must be smaller!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create carver object to store fields that need to be memoized (i.e. energy)
    // NOTE: the exact type of carver used is implementation specific, and may be a subclass.
    // Therefore, use the prepare_carver fn.
    auto energy = hpc_cuda::CudaEnergy(image.cols(), image.rows());
    auto carver = carver::Carver(&image, &energy);

    // Repeatedly vertically shrink it until it fits target width.
    while (image.cols() != new_width) {
        auto seam = carver.vertical_seam();
        carver.remove_vert_seam(seam);
    }

    // Now, repeatedly horizontally shrink until it fits target height.
    while (image.rows() != new_height) {
        auto seam = carver.horiz_seam();
        carver.remove_horiz_seam(seam);
    }

    // With image dimensions sufficiently changed, write out the target image.
    image.write_image(out_path);
}

void usage() {
    std::cout << "HPCarver Usage:" << std::endl;
    std::cout << "hpcarver [source_image.ppm] [out_image.ppm] [new width] [new height]" << std::endl;
}

namespace carver {

// ***** HORIZONTAL SEAM CALCULATORS ***** //

void Carver::horiz_energy() {
    for (auto row = 0; row < energy->rows(); ++row) {
        energy->set_energy(0, row, pixel_energy(0, row));
    }

    // Now set energy to minimum of three neighbors.
    for (auto col = 1; col < energy->cols(); ++col) {
        for (auto row = 0; row < energy->rows(); ++row) {
            // No wrapping
            auto neighbor_energies = energy->get_left_predecessors(col, row);

            // Energy = local energy + min(neighbors)
            uint32_t local_energy = pixel_energy(col, row);
            local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
            energy->set_energy(col, row, local_energy);
        }
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

    // This is one of the larger opportunities for parallelism.
    // Set energy to minimum of three above neighbors.
    for (auto row = 1; row < energy->rows(); ++row) {
        for (auto col = 0; col < energy->cols(); ++col) {
            // Note: no wrapping in seams!
            auto neighbor_energies = energy->get_top_predecessors(col, row);

            // energy = local energy + min(neighbors)
            uint32_t local_energy = pixel_energy(col, row);
            local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
            energy->set_energy(col, row, local_energy);
        }
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
