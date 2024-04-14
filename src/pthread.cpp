/**
 * Posix Thread Implementation of HPCarver. Currently, this will just spawn threads.
 * In the future, I may switch this to prepare a thread pool.
 *
 * Author: Will Morris
 */
#include <cassert>
#include <algorithm>

#include "carver.h"
#include "energy.h"

namespace carver {


// ***** THREADING METADATA ***** //

/**
 * This may be updated by an env var.
 * If not, by default use four threads.
 */
static int num_threads = 4;

/**
 * Data structure passed to a thread during pthread_create.
 *
 * @param carver pointer to carver object to perform operations on.
 * This contains both an energy and image matrix, so both may be updated based on the operation.
 *
 * @param start_row row this computation should start on.
 * In single row computations, (i.e. traversing a col), should equal end_row
 *
 * @param end_row row this computation should end on.
 * In single row computations (i.e. traversing a col), should equal start_row
 *
 * @param start_col column this computation should start on.
 * In single column computations (i.e. traversing a row), should equal end_col
 *
 * @param end_col column this computation should end on.
 * In single column computations (i.e. traversing a row), should equal start_col
 */
struct thread_data {
    // NOTE: use pointers -- otherwise, major risk of
    Carver *carver;
    uint32_t start_row;
    uint32_t end_row;
    uint32_t start_col;
    uint32_t end_col;
};

/**
 * Thread data constructor.
 * Note: either the start and end row should be the same,
 * or the start and end column should be the same.
 * If an operation traverses multiple rows and columns, will be data access issues.
 *
 * @param carver Carver object. Will be used to read energy vector.
 * @param start_row Row computation should start on.
 * @param end_row Row computation should end on.
 * @param start_col Column computation should start on.
 * @param end_col Column computation should end on.
 * @return Heap-allocated thread data structure with these params.
 */
thread_data *new_thread_data(Carver *carver,
                             uint32_t start_row, uint32_t end_row, uint32_t start_col, uint32_t end_col) {

    assert(start_row == end_row || start_col == end_col);
    auto *data = static_cast<thread_data *>(calloc(1, sizeof(thread_data)));

    data->carver = carver;
    data->start_row = start_row;
    data->end_row = end_row;
    data->start_col = start_col;
    data->end_col = end_col;

    return data;
}

// ***** HORIZONTAL ENERGY ***** //

/**
 * Threaded horizontal image energy updater.
 * Given a thread_data struct, traverse the specified column, updating each pixel's energies
 * Based on the minimum of their left neighbors.
 *
 * @param data: data structure for horizontal seam removal, contains carver object
 * @return Nothing; the changes will be made to data->carver
 */
void *update_horiz_energy(void *data);

/**
 * Horizontal energy updating entry point.
 * Traverses the carver's energy matrix from left to right, updating energy based on base_energy + predecessors.
 * @param carver Carver to perform operations on.
 */
void horiz_energy(Carver *carver);

/**
 * Find the end of the minimum horizontal seam by traversing the back column of the energy matrix.
 * Then, we will be able to find the optimal seam by traversing in reverse order.
 * @param carver Carver to consider.
 * @return Row of minimum energy pixel in last column.
 */
uint32_t min_horiz_seam_end(Carver *carver);

/**
 * Given an end index and a carver, traverse the carver's energy matrix
 * Finding the minimum connected index at each point.
 *
 * @param end_index Index at the end of the seam; this will prime the computation.
 * @return The minimum seam, in the correct direction.
 */
std::vector<uint32_t> min_horiz_seam(Carver *carver, uint32_t end_index);

// ***** VERTICAL ENERGY ***** //

/**
 * Threaded vertical image energy updater.
 * Given a thread_data struct, traverse the specified row, updating each pixel's energies
 * Based on the minimum of their upper neighbors.
 *
 * @param data: data structure for vertical seam removal, contains carver object
 * @return Nothing, changes will be made to data->carver
 */
void *update_vert_energy(void *data);

/**
 * Compute the minimum energy of each pixel in the vertical direction, storing in carver's energy memo structure.
 * @param carver Carver to mutate.
 */
void vert_energy(Carver *carver);

/**
 * Find the end of the minimum vertical seam by traversing the bottom row of the energy matrix.
 * Then, we will be able to find the optimal seam by traversing in reverse order.
 * @param carver Carver to consider.
 * @return Column of minimum energy pixel in bottom row.
 */
uint32_t min_vert_seam_end(Carver *carver);

/**
 * Given an end index and a carver, traverse the carver's energy matrix
 * Finding the minimum connected index at each point.
 *
 * @param end_index Index at the end of the seam; this will prime the computation.
 * @return The minimum seam, in the correct direction.
 */
std::vector<uint32_t> min_vert_seam(Carver *carver, uint32_t end_index);


// ***** IMPLEMENTATIONS ***** //


// ***** HORIZONTAL ENERGY UPDATERS ***** //


std::vector<uint32_t> Carver::horiz_seam() {
    assert_valid_dims();

    // Compute horizontal energies.
    horiz_energy(this);
    uint32_t back_col = energy.cols() - 1;

    // Now, prime the reverse traversal with the minimum above energy.
    auto seam_end = min_horiz_seam_end(this);
    // And return the minimum seam with this primed data.
    return min_horiz_seam(this, seam_end);
}

// ***** HORIZONTAL HELPERS ***** ///

void horiz_energy(Carver *carver) {
    auto energy = carver->get_energy();

    // Generate energy matrix
    // Horizontal seam direction: left to right.
    // Prime memo structure with base energies of first pixel column.
    for (auto row = 0; row < energy->rows(); ++row) {
        energy->set_energy(0, row, carver->pixel_energy(0, row));
    }

    // NOTE: since number of rows will not decrease during a horizontal seam update
    // Can determine stride here.
    uint32_t stride = energy->rows() / num_threads;
    pthread_t thread_pool[num_threads];

    // Now set energy to minimum of three neighbors.
    for (uint32_t col = 1; col < energy->cols(); ++col) {

        // Split work as evenly as possible between all the threads we have
        for (uint32_t thread_num = 0; thread_num < num_threads; ++thread_num) {
            uint32_t start_row = thread_num * stride;
            auto end_row = thread_num + stride;

            // Edge case: on an odd number of allocations, there may be an extra datum left over.
            // Just give it to the last thread.
            if (end_row == energy->rows() - 1) {
                end_row += 1;
            }

            auto *data = new_thread_data(carver, start_row, end_row, col, col);
            pthread_create(&thread_pool[thread_num], nullptr,
                           update_horiz_energy, (void *) data);
            // Data will be freed by thread when it's done.
        }

        // For now, not using a job queue -- spawning, reaping threads on each iteration.
        for (auto thread : thread_pool) {
            pthread_join(thread, nullptr);
        }
    }
}

void *update_horiz_energy(void *data1) {
    auto *data = (thread_data*) data1;

    // Note: these should all be pointers, not references.
    // C++ does weird automatic deallocation stuff when you use references.
    auto carver = data->carver;
    auto energy = carver->get_energy();

    // Get fixed column for this data -- should be a single col.
    assert(data->start_col == data->end_col);
    assert(data->start_col >= 0 && data->start_col < energy->cols());
    auto col = data->start_col;

    // Get start, end rows for this data.
    // NOTE: currently, asserting start, end aren't equal.
    assert(data->start_row >= 0 && data->start_row < data->end_row);
    // NOTE: since end_row is an exclusive boundary, it's OK for it to equal energy->rows()
    assert(data->end_row <= energy->rows());
    auto start_row = data->start_row;
    auto end_row = data->end_row;

    // Now, perform the update over a single column.
    for (auto row = start_row; row < end_row; ++row) {
        // No wrapping
        auto neighbor_energies = energy->get_left_predecessors(col, row);

        // Energy = local energy + min(neighbors)
        // TODO: look into narrowing conversion
        uint32_t local_energy = carver->pixel_energy(col, row);
        local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
        energy->set_energy(col, row, local_energy);
    }

    free(data);
    data = nullptr;
    pthread_exit(nullptr);
}

uint32_t min_horiz_seam_end(Carver *carver) {
    auto energy = carver->get_energy();
    uint32_t back_col = energy->cols() - 1;

    // Default: row 0 of the last column contains the minimum energy.
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

    return min_row;
}

std::vector<uint32_t> min_horiz_seam(Carver *carver, uint32_t end_index) {
    auto energy = carver->get_energy();
    uint32_t back_col = energy->cols() - 1;

    // Now, prime the reverse traversal with the minimum above energy.
    auto seam = std::vector<uint32_t>{};
    seam.push_back(end_index);

    uint32_t min_row;
    uint32_t min_energy;

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


// ***** VERTICAL SEAM IMPLEMENTATIONS ***** //

void *update_vert_seam(void *data1) {
    auto *data = (thread_data*) data1;
    auto carver = data->carver;
    auto energy = carver->get_energy();

    // Get fixed row for this data -- should be a single col.
    assert(data->start_row == data->end_row);
    assert(data->start_row >= 0 && data->start_row < carver->get_energy()->rows());
    auto row = data->start_row;

    // Get start, end rows for this data.
    // NOTE: currently, asserting start, end aren't equal.
    assert(data->start_col >= 0 && data->start_col < data->end_col);
    assert(data->end_col < carver->get_energy()->cols());
    auto start_col = data->start_row;
    auto end_col = data->end_row;

    // Now, perform the update over a single row.
    for (auto col = start_col; col < end_col; ++col) {
        // No wrapping
        auto neighbor_energies = energy->get_top_predecessors(col, row);

        // Energy = local energy + min(neighbors)
        // TODO: look into narrowing conversion
        uint32_t local_energy = carver->pixel_energy(col, row);
        local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
        energy->set_energy(col, row, local_energy);
    }

    pthread_exit(nullptr);
}


// ***** STANDARD CARVER IMPLS ***** //

// Carver constructor
Carver::Carver(hpimage::Hpimage &image):
    image(image), energy(Energy(image.cols(), image.rows())) {
    assert_valid_dims();
    // TODO: look into getenv to update num_threads
}


std::vector<uint32_t> Carver::vertical_seam() {
    assert_valid_dims();

    // Vertical seam direction: top to bottom
    // Prime memo structure with base energies of first pixel row.
    for (auto col = 0; col < energy.cols(); ++col) {
        energy.set_energy(col, 0, pixel_energy(col, 0));
    }

    // This is one of the larger opportunities for parallelism.
    // Set energy to minimum of three above neighbors.
    for (auto row = 1; row < energy.rows(); ++row) {
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

void Carver::remove_horiz_seam(std::vector<uint32_t> &seam) {
    // Must be exactly one row to remove from each column.
    assert(seam.size() == image.cols());

    for (auto col = 0; col < image.cols(); ++col) {
        auto index = seam[col];
        assert(index >= 0 && index < image.rows());

        // Shift all pixels below this up one.
        for (auto row = index; row < image.rows() - 1; ++row) {
            hpimage::pixel below = image.get_pixel(col, row + 1);
            image.set_pixel(col, row, below);
        }
    }
    // Finally, cut the last row from the pixel.
    energy.cut_row();
    image.cut_row();
}

void Carver::remove_vert_seam(std::vector<uint32_t> &seam) {
    // Must be exactly one column to remove from each row.
    assert(seam.size() == image.rows());

    // Shift every pixel after a given image over.
    // Then reduce image size by one.
    for (auto row = 0; row < image.rows(); ++row) {
        auto index = seam[row];
        assert(index >= 0 && index < image.cols());

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
