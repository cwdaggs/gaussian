#include <limits.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void gaussian_calc(unsigned char *image_mat, float *kernel, int width, int height, float order) 
{
    float val = 0;

    int center = ((int) order - 1) / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int x = 0; x < (int) order; x++) {
                for (int y = 0; y < (int) order; y++) {
                    if (i < center && j < center) { //top left corner
                        if (x < center && y < center) {
                            val += (image_mat[i * height + j] * kernel[x * (int) order + y]);
                        } else if (x < center && y > center) {
                            val += (image_mat[i * height + j + (y - center)] * kernel[x * (int) order + y]);
                        } else if (y < center && x > center) {
                            val += (image_mat[(i + (x - center)) * height + j] * kernel[x * (int) order + y]);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                        }
                    } else if (i < center && (j >= center && j <= width - center - 1)) { //top edge
                        if (x < center - i) {
                            val += (image_mat[i * height + j + (y - center)] * kernel[x * (int) order + y]);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                        }
                    } else if (i > height - center - 1 && (j >= center && j <= width - center - 1)) { //bottom edge
                        if (i + x > height + center - 1) {
                            val += (image_mat[i * height + j + (y - center)] * kernel[x * (int) order + y]);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                        }
                    } else if (j < center && (i >= center && i <= height - center - 1)) { //left edge
                        if (y < center - j) {
                            val += (image_mat[(i + (x - center)) * height + j] * kernel[x * (int) order + y]);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                        }
                    } else if (j > width - center - 1 && (i >= center && i <= height - center - 1)) { //right edge
                        if (j + y > width + center - 1) {
                            val += (image_mat[(i + (x - center)) * height + j] * kernel[x * (int) order + y]);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                        }
                    } else if (j > width - center && i < center) { //top right corner
                        continue;
                    } else if (i > height - center && j < center) { // bottom left corner
                        continue;
                    } else if (i > height - center && j > width - center) { //bottom right corner
                        continue;
                    } else {
                        val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                    }
                }
            }
            image_mat[i * height + j] = (unsigned char) val; 
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
	fprintf(fp, "P5\n%d %d\n255\n", width, height);

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
    output_filename = argv[2];

    if (fscanf(input_file, "%*[^\n]\n") != 0) {
        exit(1);
    }
    if (fscanf(input_file, "%d %d\n", &width, &height) != 2) {
        exit(1);
    }
    if (fscanf(input_file, "%*[^\n]\n") != 0) {
        exit(1);
    }
    
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

    if (fread(image_mat, sizeof(unsigned char), height * width, input_file) != (size_t)(height * width)) {
        exit(1);
    }
    fclose(input_file);

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
    gaussian_calc(image_mat, kernel, width, height, order);

	write_mandelmap(output_filename, image_mat, width, height);
    
	// free(filename);

	return 0;
}