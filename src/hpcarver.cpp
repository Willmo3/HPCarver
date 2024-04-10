// Entry file for HPcarver
// Author: Will Morris

#include "RAJA/RAJA.hpp"
#include "../HPImage/hpimage.h"
#include "carver.h"

// Print usage data
void usage();

// Indexes for argument parsing
#define ARG_COUNT 5
#define SOURCE_INDEX 1
#define TARGET_INDEX 2
#define WIDTH_INDEX 3
#define HEIGHT_INDEX 4

// ***** PROGRAM ENTRY POINT ***** //

int main(int argc, char *argv[]) {
    // For now, default to sequential execution
    using policy = RAJA::seq_exec;

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
    auto image = hpimage::Hpimage(source_path);

    // Ensure that image is being shrunk -- for now, this is all that's supported!
    if (new_width > image.cols() || new_height > image.rows()) {
        std::cerr << "ERROR: HPCarver supports shrinking. New image dimensions must be smaller!" << std::endl;
    }

    // Repeatedly vertically shrink it until it fits target width.
    while (image.cols() != new_width) {
        auto seam = carver::vertical_seam(image);
        carver::remove_vert_seam(image, *seam);
    }

    // Now, repeatedly horizontally shrink until it fits target height.
    while (image.rows() != new_height) {
        auto seam = carver::horiz_seam(image);
        carver::remove_horiz_seam(image, *seam);
    }

    // With image dimensions sufficiently changed, write out the target image.
    image.write_image(out_path);
}

void usage() {
    std::cout << "HPCarver Usage:" << std::endl;
    std::cout << "hpcarver [source_image.ppm] [out_image.ppm] [new width] [new height]" << std::endl;
}