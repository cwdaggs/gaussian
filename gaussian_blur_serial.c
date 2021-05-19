#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M_PI acos(-1.0)

void gaussian_calc(unsigned char *image_mat, float *kernel, int width, int height, float order) 
{
    int val = 0;

    int center = ((int) order - 1) / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int x = 0; x < (int) order; x++) {
                for (int y = 0; y < (int) order; y++) {
                    if (i < center || j < center) {
                        if (x <= center && y <= center) {
                            val += (image_mat[i * (int) order + j] * kernel[x * (int) order + y]);
                        } else if (x < center && y > center) {
                            val += (image_mat[i * (int) order + j + (y - center)] * kernel[x * (int) order + y]);
                        } else if (y < center && x > center) {
                            val += (image_mat[(i + (x - center)) * (int) order + j] * kernel[x * (int) order + y]);
                        }
                    }
                }
            }
            image_mat[i * (int) order + j] = val; 
            val = 0;
        }
    }
}

void write_mandelmap(char *filename, unsigned char *mandelmap, int width, int height)
{
	FILE *fp;

	/* Open file */
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Error: cannot open file %s", filename);
		exit(1);
	}

	/* Put structural information */
	fprintf(fp, "P5\n%ld %ld\n255\n", width, height);

	/* Output grayscale pixels */
	fwrite(mandelmap, sizeof(unsigned char), width * height, fp);

	// free(pixels);
	fclose(fp);
}

int main(int argc, char *argv[])
{
	float sigma, order;
	char *output_filename;
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
    if ((int)order % 2 == 0) {
        order++;
    }
    if (order > width || order > height) {
        fprintf(stderr, "Error: sigma value too big for image size");
		exit(1);
    }

    float *kernel = aligned_alloc(64, (int) order * (int) order * sizeof(float));
    unsigned char *image_mat = aligned_alloc(64, width * height * sizeof(unsigned char));

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            fread(image_mat[i * width + j], sizeof(unsigned char), 1, input_file);
        }
    } 

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            kernel[i * (int) order + j] = (1/(2*M_PI*sigma*sigma)) * 
            exp(-(pow(i - floor(order/2), 2) + pow(j - floor(order/2), 2))/(2 * sigma * sigma));
        }
    }

	/* Align allocation on a cache line (64 bytes) */
	// mandelmap = aligned_alloc(64, N * N * sizeof(char));

	// /* !!! Calling your implementation !!! */
	// mandel_calc(mandelmap, N, x_coord, y_coord, zoom_level, cutoff);

	// /* Save output image */
	write_mandelmap(output_filename, image_mat, width, height);
    fclose(input_file);
	// free(filename);

	return 0;
}