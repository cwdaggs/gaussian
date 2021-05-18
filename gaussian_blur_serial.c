#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void write_mandelmap(char *filename, unsigned char *mandelmap, size_t N, int cutoff)
{
	FILE *fp;

	/* Open file */
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Error: cannot open file %s", filename);
		exit(1);
	}

	/* Put structural information */
	fprintf(fp, "P5\n%ld %ld\n%d\n", N, N, cutoff);

	/* Output grayscale pixels */
	fwrite(mandelmap, sizeof(unsigned char), N * N, fp);

	// free(pixels);
	fclose(fp);
}

int main(int argc, char *argv[])
{
	float sigma, order;
	char *input_filename, output_filename;
    int width, height;
    FILE *input_file;

	/* Command line arguments */
	if (argc < 4) {
		fprintf(stderr, "Usage: %s <input_pgm> <output_pgm> <sigma>\n",
				argv[0]);
		exit(1);
	}

    input_file = fopen(argv[1], "rb");
    if (!input_file) {
        fprintf(stderr, "Error: cannot open file %s", argv[1]);
		exit(1);
    }

    fscanf(input_file, "%s", NULL);
    fscanf(input_file, "%d %d", width, height);
    sigma = atoi(argv[3]);
    if (sigma <= 0) {
        fprintf(stderr, "Error: invalid sigma value");
		exit(1);
    }
    order = ceil(6 * sigma);
    if (order > width || order > height) {
        fprintf(stderr, "Error: sigma value too big for image size");
		exit(1);
    }


	/* Align allocation on a cache line (64 bytes) */
	// mandelmap = aligned_alloc(64, N * N * sizeof(char));

	// /* !!! Calling your implementation !!! */
	// mandel_calc(mandelmap, N, x_coord, y_coord, zoom_level, cutoff);

	// /* Save output image */
	// filename = malloc(PATH_MAX);
	// sprintf(filename, "mandel_%d_%.3lf_%.3lf_%.3lf_%d",
	// 		N, x_coord, y_coord, zoom_level, cutoff);
	// if (argc > 6)
	// 	strcat(filename, argv[6]);
	// strcat(filename, ".pgm");
	// write_mandelmap(filename, mandelmap, N, cutoff);
	// free(filename);

	return 0;
}