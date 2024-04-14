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
    // In case we need to pass seams around
    std::vector<uint32_t> *seam;
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
 * @param seam (Optional) Pointer to seam to remove.
 * @param start_row Row computation should start on.
 * @param end_row Row computation should end on.
 * @param start_col Column computation should start on.
 * @param end_col Column computation should end on.
 * @return Heap-allocated thread data structure with these params.
 */
thread_data *new_thread_data(Carver *carver, std::vector<uint32_t> *seam,
                             uint32_t start_row, uint32_t end_row, uint32_t start_col, uint32_t end_col) {

    assert(start_row == end_row || start_col == end_col);
    auto *data = static_cast<thread_data *>(calloc(1, sizeof(thread_data)));

    data->carver = carver;
    data->seam = seam;
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
        uint32_t local_energy = carver->pixel_energy(col, row);
        local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
        energy->set_energy(col, row, local_energy);
    }

    free(data);
    data = nullptr;
    pthread_exit(nullptr);
}

/**
 * Threaded vertical image energy updater.
 * Given a thread_data struct, traverse the specified row, updating each pixel's energies
 * Based on the minimum of their upper neighbors.
 *
 * @param data: data structure for vertical seam removal, contains carver object
 * @return Nothing, changes will be made to data->carver
 */
void *update_vert_energy(void *data1) {
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
    assert(data->end_col <= energy->cols());
    auto start_col = data->start_col;
    auto end_col = data->end_col;

    // Now, perform the update over a single row.
    for (auto col = start_col; col < end_col; ++col) {
        // No wrapping
        auto neighbor_energies = energy->get_top_predecessors(col, row);

        // Energy = local energy + min(neighbors)
        uint32_t local_energy = carver->pixel_energy(col, row);
        local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
        energy->set_energy(col, row, local_energy);
    }

    free(data);
    data = nullptr;
    pthread_exit(nullptr);
}

/**
 * Given a seam and a set of columns representing portion of seam to operate on.
 * shift all rows after the seam up by one.
 *
 * @param data1 Data structure containing seam and columns to shift up.
 */
void *shift_horiz(void *data1) {
    auto data = (thread_data *) data1;

    auto image = data->carver->get_image();
    // Horizontal seam removal's rows are defined by the seam.
    // So rows in this data structure ought to be 0.
    assert(data->start_row == 0 && data->end_row == 0);
    assert(data->start_col >= 0 && data->end_col <= image->cols());

    for (auto col = data->start_col; col < data->end_col; ++col) {
        auto index = data->seam->at(col);
        assert(index >= 0 && index < image->rows());

        // Shift all pixels below this up one.
        for (auto row = index; row < image->rows() -1; ++row) {
            hpimage::pixel below = image->get_pixel(col, row + 1);
            image->set_pixel(col, row, below);
        }
    }

    free(data);
    data = nullptr;
    pthread_exit(nullptr);
}

/**
 * Given a seam and a set of rows representing portion of seam to operate on.
 * shift all columns after the seam up by one.
 *
 * @param data1 Data structure containing seam and columns to shift up.
 */
void *shift_vert(void *data1) {
    auto data = (thread_data *) data1;

    auto image = data->carver->get_image();
    // Horizontal seam removal's rows are defined by the seam.
    // So rows in this data structure ought to be 0.
    assert(data->start_col == 0 && data->end_col == 0);
    assert(data->start_row >= 0 && data->end_col <= image->rows());

    for (auto row = data->start_row; row < data->end_row; ++row) {
        auto index = data->seam->at(row);
        assert(index >= 0 && index < image->rows());

        // Shift all pixels to the right of this one left.
        for (auto col = index; col < image->cols() -1; ++col) {
            hpimage::pixel right = image->get_pixel(col + 1, row);
            image->set_pixel(col, row, right);
        }
    }

    free(data);
    data = nullptr;
    pthread_exit(nullptr);
}


// ***** HORIZONTAL SEAM CALCULATORS ***** //

void Carver::horiz_energy() {
    // Generate energy matrix
    // Horizontal seam direction: left to right.
    // Prime memo structure with base energies of first pixel column.
    for (auto row = 0; row < energy.rows(); ++row) {
        energy.set_energy(0, row, pixel_energy(0, row));
    }

    uint32_t stride = energy.rows() / num_threads;
    // Round up work.
    // Then, the last thread will do less work.
    if (energy.rows() % num_threads) {
        stride += 1;
    }
    pthread_t thread_pool[num_threads];


    // Now set energy to minimum of three neighbors.
    for (uint32_t col = 1; col < energy.cols(); ++col) {
        uint32_t thread_num = 0;
        uint32_t start_row = 0;
        uint32_t end_row = start_row + stride;

        while (thread_num < num_threads && start_row < energy.rows()) {
            if (end_row > energy.rows()) {
                // Edge case: we've run out of work for the threads!
                end_row = energy.rows();
            }

            auto *data = new_thread_data(this, nullptr, start_row, end_row, col, col);
            pthread_create(&thread_pool[thread_num], nullptr,
                           update_horiz_energy, (void *) data);

            ++thread_num;
            start_row += stride;
            end_row += stride;
        }

        // Join all the threads that we spawned.
        // For now, not using job queue.
        for (auto thread = 0; thread < thread_num; ++thread) {
            pthread_join(thread_pool[thread], nullptr);
        }
    }
}

std::vector<uint32_t> Carver::min_horiz_seam() {
    uint32_t back_col = energy.cols() - 1;

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

    // Now, prime the reverse traversal with the minimum above energy.
    auto seam = std::vector<uint32_t>{};
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
    // Prime memo structure with base energies of first pixel row.
    for (auto col = 0; col < energy.cols(); ++col) {
        energy.set_energy(col, 0, pixel_energy(col, 0));
    }

    uint32_t stride = energy.cols() / num_threads;

    // If work allocation doesn't fit perfectly, assign slightly more work to each thread.
    // Then, they can each do a little extra.
    if (energy.cols() % num_threads) {
        stride += 1;
    }
    pthread_t thread_pool[num_threads];


    // Set energy to minimum of three above neighbors.
    for (auto row = 1; row < energy.rows(); ++row) {
        uint32_t thread_num = 0;
        uint32_t start_col = 0;
        uint32_t end_col = start_col + stride;

        while (thread_num < num_threads && start_col < energy.cols()) {
            // Edge case: odd amount of work, give the last tasks to the final thread.
            if (end_col > energy.cols()) {
                end_col = energy.cols();
            }

            auto *data = new_thread_data(this, nullptr, row, row, start_col, end_col);
            pthread_create(&thread_pool[thread_num], nullptr,
                           update_vert_energy, (void *) data);

            // Update for next iteration
            ++thread_num;
            start_col += stride;
            end_col += stride;
        }

        // For now, not using a job queue -- spawning, reaping threads on each iteration.
        // Only traverse through the thread numbers that we spawned
        for (uint32_t thread = 0; thread < thread_num; ++thread) {
            pthread_join(thread_pool[thread], nullptr);
        }
    }
}

std::vector<uint32_t> Carver::min_vert_seam() {
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


// ***** SEAM REMOVERS ***** //

void Carver::remove_horiz_seam(std::vector<uint32_t> &seam) {
    // Must be exactly one row to remove from each column.
    assert(seam.size() == image.cols());

    // Must compute stride size.
    pthread_t thread_pool[num_threads];
    uint32_t stride = image.cols() / num_threads;

    // Edge case: too little work for all the threads we want!
    if (stride == 0) {
        stride = 1;
    }

    uint32_t thread_num = 0;
    uint32_t start_col = 0;
    uint32_t end_col = start_col + stride;

    while (thread_num < num_threads && end_col <= image.cols()) {
        if (thread_num == num_threads - 1 && end_col != image.cols()) {
            end_col = image.cols();
        }

        auto *data = new_thread_data(this, &seam, 0, 0, start_col, end_col);
        pthread_create(&thread_pool[thread_num], nullptr, shift_horiz, data);

        ++thread_num;
        start_col += stride;
        end_col += stride;
    }

    for (auto thread = 0; thread < thread_num; ++thread) {
        pthread_join(thread_pool[thread], nullptr);
    }

    // Finally, cut the last row from the pixel.
    energy.cut_row();
    image.cut_row();
}

void Carver::remove_vert_seam(std::vector<uint32_t> &seam) {
    // Must be exactly one column to remove from each row.
    assert(seam.size() == image.rows());

    // Must compute stride size.
    pthread_t thread_pool[num_threads];
    uint32_t stride = image.rows() / num_threads;

    // Edge case: too little work for all the threads we want!
    if (stride == 0) {
        stride = 1;
    }

    uint32_t thread_num = 0;
    uint32_t start_row = 0;
    uint32_t end_row = start_row + stride;

    while (thread_num < num_threads && end_row <= image.rows()) {
        if (thread_num == num_threads - 1 && end_row != image.rows()) {
            end_row = image.rows();
        }

        auto *data = new_thread_data(this, &seam, start_row, end_row, 0, 0);
        pthread_create(&thread_pool[thread_num], nullptr, shift_vert, data);

        ++thread_num;
        start_row += stride;
        end_row += stride;
    }

    for (auto thread = 0; thread < thread_num; ++thread) {
        pthread_join(thread_pool[thread], nullptr);
    }

    // Finally, with all pixels shifted over, time to trim the image!
    energy.cut_col();
    image.cut_col();
}

// Carver constructor
Carver::Carver(hpimage::Hpimage &image):
    image(image), energy(Energy(image.cols(), image.rows())) {
    assert_valid_dims();
    if (const char *thread_env = getenv("HPC_THREADS")) {
        num_threads = std::stoi(thread_env);
    }
}

} // namespace carver
