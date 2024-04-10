// Entry file for HPcarver
// Author: Will Morris

#include <getopt.h>
#include "RAJA/RAJA.hpp"
#include "../HPImage/hpimage.h"

// Print usage data
void usage();

#define ARG_COUNT 9

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

    // Load image
    auto image = hpimage::Hpimage(source_path);

    // Ensure that image is being shrunk -- for now, this is all that's supported!
    if (new_width > image.cols() || new_height > image.rows()) {
        std::cerr << "ERROR: HPCarver supports shrinking. New image dimensions must be smaller!" << std::endl;
    }
    // Repeatedly horizontally shrink it until it fits target width.

    // Repeatedly vertically shrink it until it fits target height.

    // Write out new image.

}


// ***** ARGUMENT MANAGERS ***** //

void parse_args(int argc, char *argv[],
        char *&source_path, char *&out_path, uint32_t &new_width, uint32_t &new_height) {
    // There should be "9" options.
    if (argc != ARG_COUNT) {
        usage();
        exit(EXIT_FAILURE);
    }

    // Parse all arguments
    // Getopt parsing derived from https://www.gnu.org/software/libc/manual/html_node/Example-of-Getopt.html
    int c;
    while ((c = getopt(argc, argv, "i:o:w:h:")) != -1) {
        switch(c) {
            case 'i':
                source_path = optarg;
                break;
            case 'o':
                out_path = optarg;
                break;
            case 'w':
                new_width = std::stoi(optarg);
                break;
            case 'h':
                new_height = std::stoi(optarg);
                break;
            default:
                std::cout << "Unrecognized option: " << (char) c << std::endl;
                usage();
                exit(EXIT_FAILURE);
        }
    }

    // No arguments are optional -- error if anything missing!
    if (!source_path) {
        std::cout << "ERROR: Must specify image source path!" << std::endl;
        usage();
        exit(EXIT_FAILURE);
    }
    if (!out_path) {
        std::cout << "ERROR: Must specify image output path!" << std::endl;
        usage();
        exit(EXIT_FAILURE);
    }
    // HPCarver only supports width < three.
    if (new_width < 3) {
        std::cout << "ERROR: Must specify new image width greater than three!" << std::endl;
        usage();
        exit(EXIT_FAILURE);
    }
    if (new_height < 3) {
        std::cout << "ERROR: Must specify new image height greater than three!" << std::endl;
        usage();
        exit(EXIT_FAILURE);
    }
}

void usage() {
    std::cout << "HPCarver Usage:" << std::endl;
    std::cout << "hpcarver -i [source_image.ppm] -o [out_image.ppm] -w [new width] -h [new height]" << std::endl;
}