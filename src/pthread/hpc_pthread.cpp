// Entry file for hpc_pthread
// Author: Will Morris

#include "../../HPImage/hpimage.h"
#include "../carver/carver.h"
#include "../carver/timer.h"
#include "pthread.h"

// Print usage data
void usage();

// Indexes for argument parsing
#define ARG_COUNT 6
#define THREAD_INDEX 1
#define SOURCE_INDEX 2
#define TARGET_INDEX 3
#define WIDTH_INDEX 4
#define HEIGHT_INDEX 5

// ***** PROGRAM ENTRY POINT ***** //

int main(int argc, char *argv[]) {
    // For now, default to sequential execution

    // There should be 5 options, including executable name
    if (argc != ARG_COUNT) {
        usage();
        exit(EXIT_FAILURE);
    }

    int threads = std::stoi(argv[THREAD_INDEX]);
    if (threads < 0) {
        std::cerr << "ERROR: Negative thread count: " << threads << " specified." << std::endl;
        exit(EXIT_FAILURE);
    }

    hpc_pthread::hpc_pthread_init(threads);

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
    auto image = hpimage::Hpimage(source_path);

    // Ensure that image is being shrunk -- for now, this is all that's supported!
    if (new_width > image.cols() || new_height > image.rows()) {
        std::cerr << "ERROR: HPCarver supports shrinking. New image dimensions must be smaller!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create carver object to store fields that need to be memoized (i.e. energy)
    // NOTE: the exact type of carver used is implementation specific, and may be a subclass.
    // Therefore, use the prepare_carver fn.
    auto energy = carver::Energy(image.cols(), image.rows());
    auto carver = carver::Carver(&image, &energy);

    auto timer = carver::Timer();

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

    std::cout << "HPC pthread carving time:" << std::endl;
    std::cout << timer.elapsed() << std::endl;
    std::cout << "(num threads: " << threads << ")" << std::endl;

    // With image dimensions sufficiently changed, write out the target image.
    image.write_image(out_path);
}

void usage() {
    std::cout << "HPCarver Pthread Usage:" << std::endl;
    std::cout << "hpc_pthread [num_threads] [source_image.ppm] [out_image.ppm] [new width] [new height]" << std::endl;
}
