// Entry file for HPcarver
// Author: Will Morris

#include <getopt.h>
#include "RAJA/RAJA.hpp"

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

    char *source_path;
    char *target_path;
    uint32_t new_width;
    uint32_t new_height;

    parse_args(argc, argv, source_path, target_path, new_width, new_height);
}


// ***** ARGUMENT MANAGERS ***** //

void parse_args(int argc, char *argv[],
        char *&source_path, char *&out_path, uint32_t &new_width, uint32_t &new_height) {
    // No optional arguments
    if (argc != ARG_COUNT) {
        usage();
        exit(EXIT_FAILURE);
    }

    int c;

    // Getopt parsing derived from https://www.gnu.org/software/libc/manual/html_node/Example-of-Getopt.html
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
}

void usage() {
    std::cout << "HPCarver Usage:" << std::endl;
    std::cout << "hpcarver -i [source_image.ppm] -o [out_image.ppm] -w [new width] -h [new height]" << std::endl;
}