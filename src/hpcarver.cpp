// Entry file for HPcarver
// Author: Will Morris

#include "RAJA/RAJA.hpp"
#include "../HPImage/hpimage.h"
#include "carver.h"

// Print usage data
void usage();

#define ARG_COUNT 5
#define SOURCE_INDEX 1
#define TARGET_INDEX 2
#define WIDTH_INDEX 3
#define HEIGHT_INDEX 4

/**
 * Parse arguments for necessary fields. Note that all of these are required!
 *
 * @param argc Number of arguments. Should be 5 -- one extra argument for program name.
 * @param argv Vector of arguments
 * @param source_path Path of source image to convert.
 * @param out_path Path to image to output
 * @param new_width Width of new image. Should be lower than input image -- we currently only support scaling down.
 * @param new_height Height of new image.
 */
void parse_args(int argc, char *argv[],
        char *&source_path, char *&out_path, uint32_t &new_width, uint32_t &new_height);


// ***** PROGRAM ENTRY POINT ***** //

int main(int argc, char *argv[]) {
    // For now, default to sequential execution
    using policy = RAJA::seq_exec;

    char *source_path = nullptr;
    char *target_path = nullptr;
    // Initialized to negative number as param validation.
    uint32_t new_width = 0;
    uint32_t new_height = 0;

    parse_args(argc, argv, source_path, target_path, new_width, new_height);

    // HPCarver only supports width < three.
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
    image.write_image(target_path);
}


// ***** ARGUMENT MANAGERS ***** //

void parse_args(int argc, char *argv[],
        char *&source_path, char *&out_path, uint32_t &new_width, uint32_t &new_height) {
    // There should be "9" options.
    if (argc != ARG_COUNT) {
        usage();
        exit(EXIT_FAILURE);
    }

    source_path = argv[SOURCE_INDEX];
    out_path = argv[TARGET_INDEX];
    new_width = std::stoi(argv[WIDTH_INDEX]);
    new_height = std::stoi(argv[HEIGHT_INDEX]);
}

void usage() {
    std::cout << "HPCarver Usage:" << std::endl;
    std::cout << "hpcarver [source_image.ppm] [out_image.ppm] [new width] [new height]" << std::endl;
}