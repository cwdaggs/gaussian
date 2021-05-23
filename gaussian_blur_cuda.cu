#include <limits.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define cuda_check(ret) _cuda_check((ret), __FILE__, __LINE__)
inline void _cuda_check(cudaError_t ret, const char *file, int line) {
    if (ret != cudaSuccess) {
        fprintf(stderr, "CudaErr: %s (%s:%d)\n", cudaGetErrorString(ret), file, line);
        exit(1);    
    }
}



__global__ void gaussian_calc_kernel(unsigned char *image_mat, unsigned char *result_mat, float *kernel, int width, int height, float order) 
{
    float val = 0;

    int center = ((int) order - 1) / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int x = 0; x < (int) order; x++) {
                for (int y = 0; y < (int) order; y++) {
                    if (i < center && j < center) { //top left corner
                        if (x <= center - i && y <= center - j) {
                            val += (image_mat[0] * kernel[x * (int) order + y]);
                            // printf("1: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[0], x, y, i, j);
                        } else if (x < center - i && y > center - j) {
                            val += (image_mat[j + (y - center)] * kernel[x * (int) order + y]);
                            // printf("2: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[j + (y - center)], x, y, i, j);
                        } else if (y < center - j && x > center - i) {
                            val += (image_mat[(i + (x - center)) * height] * kernel[x * (int) order + y]);
                            // printf("3: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[(i + (x - center)) * height], x, y, i, j);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                            // printf("4: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[(i + (x - center)) * height + j + (y - center)], x, y, i, j);
                        }
                    } else if (i < center && (j >= center && j <= width - center - 1)) { //top edge
                        
                        if (x < center - i) {
                            // printf("entered top edge");
                            val += (image_mat[j + (y - center)] * kernel[x * (int) order + y]);
                            // if (i == 2 && j == 10) {
                            //     printf("1: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[j + (y - center)], x, y, i, j);
                            // }
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                            // if (i == 2 && j == 10) {
                            //     printf("2: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[(i + (x - center)) * height + j + (y - center)], x, y, i, j);
                            // }
                        }
                    } else if (i > height - center - 1 && (j >= center && j <= width - center - 1)) { //bottom edge
                       
                        if (i + x > height + center - 1) {
                            // printf("entered bottom edge");
                            val += (image_mat[height * (height - 1) + j + (y - center)] * kernel[x * (int) order + y]);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                        }
                    } else if (j < center && (i >= center && i <= height - center - 1)) { //left edge
                        
                        if (y < center - j) {
                            // printf("entered left edge");
                            val += (image_mat[(i + (x - center)) * height] * kernel[x * (int) order + y]);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                        }
                    } else if (j > width - center - 1 && (i >= center && i <= height - center - 1)) { //right edge
                        
                        if (j + y > width + center - 1) {
                            // printf("entered right edge");
                            val += (image_mat[(i + (x - center)) * height + width - 1] * kernel[x * (int) order + y]);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                        }
                    } else if (j > width - center - 1 && i < center) { //top right corner
                         if (x <= center - i && j + y >= width + center - 1) {
                            val += (image_mat[width - 1] * kernel[x * (int) order + y]);
                            // printf("1: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[width - 1], x, y, i, j);
                        } else if (x < center - i && j + y < width + center - 1) {
                            val += (image_mat[j + (y - center)] * kernel[x * (int) order + y]);
                            // printf("2: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[j + (y - center)], x, y, i, j);
                        } else if (j + y >= width + center - 1 && x > center - i) {
                            val += (image_mat[(i + (x - center)) * height + width - 1] * kernel[x * (int) order + y]);
                            // printf("3: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[(i + (x - center)) * height + width - 1], x, y, i, j);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                            // printf("4: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[(i + (x - center)) * height + j + (y - center)], x, y, i, j);
                        }
                    } else if (i > height - center - 1 && j < center) { // bottom left corner
                        if (i + x >= height + center - 1 && y <= center - j) {
                            val += (image_mat[height * (height - 1)] * kernel[x * (int) order + y]);
                            // printf("1: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[height * (height - 1)], x, y, i, j);
                        } else if (i + x > height + center - 1 && y > center - j) {
                            val += (image_mat[height * (height - 1) + j + (y - center)] * kernel[x * (int) order + y]);
                            // printf("2: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[height * (height - 1) + j + (y - center)], x, y, i, j);
                        } else if (y < center - j && i + x < height + center - 1) {
                            val += (image_mat[(i + (x - center)) * height] * kernel[x * (int) order + y]);
                            // printf("3: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[(i + (x - center)) * height], x, y, i, j);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                            // printf("4: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[(i + (x - center)) * height + j + (y - center)], x, y, i, j);
                        }
                    } else if (i > height - center - 1 && j > width - center - 1) { //bottom right corner
                        if (i + x >= height + center - 1 && j + y >= width + center - 1) {
                            val += (image_mat[height * (height - 1) + width - 1] * kernel[x * (int) order + y]);
                            // printf("1: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[height * (height - 1) + width - 1], x, y, i, j);
                        } else if (i + x > height + center - 1 && j + y < width + center - 1) {
                            val += (image_mat[height * (height - 1) + j + (y - center)] * kernel[x * (int) order + y]);
                            // printf("2: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[j + (y - center)], x, y, i, j);
                        } else if (j + y > width + center - 1 && i + x < height + center - 1) {
                            val += (image_mat[(i + (x - center)) * height + width - 1] * kernel[x * (int) order + y]);
                            // printf("3: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[(i + (x - center)) * height + width - 1], x, y, i, j);
                        } else {
                            val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                            // printf("4: Using %i for [%d][%d]: ij[%d][%d]\n", image_mat[(i + (x - center)) * height + j + (y - center)], x, y, i, j);
                        }
                    } else {
                        val += (image_mat[(i + (x - center)) * height + j + (y - center)] * kernel[x * (int) order + y]);
                    }
                }
            }
            result_mat[i * height + j] = (unsigned char) val; 
            val = 0;
        }
    }
}

void gaussian_calc(unsigned char *image_mat, unsigned char *result_mat, float *kernel, int width, int height, float order) 
{
    /* Allocate device memory for all matrices */
    float *kernel_d;
    unsigned char *image_mat_d, *result_mat_d;
    cuda_check(cudaMalloc(&kernel_d, (int) order * (int) order * sizeof(float)));
    cuda_check(cudaMalloc(&image_mat_d, width * height * sizeof(unsigned char)));
    cuda_check(cudaMalloc(&result_mat_d, width * height * sizeof(unsigned char)));
    
    /* Copy kernel and image_mat to device */
    cuda_check(cudaMemcpy(kernel_d, kernel, (int) order * (int) order * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(image_mat_d, image_mat, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    /* Invoke kernel function */
    dim3 grid_dim(1);
    dim3 block_dim(32, 32);
    gaussian_calc_kernel<<<grid_dim, block_dim>>>(image_mat, result_mat, kernel, width, height, order);

    /* Copy result back to host */
    cuda_check(cudaMemcpy(result_mat, result_mat_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    /* Free device memory */
    cuda_check(cudaFree(kernel_d));
    cuda_check(cudaFree(image_mat_d));
    cuda_check(cudaFree(result_mat_d));
}

void write_gaussian(char *filename, unsigned char *picture, int width, int height)
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
	fwrite(picture, sizeof(unsigned char), width * height, fp);

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

    float *kernel;
    unsigned char *image_mat, *result_mat;
    kernel = (float*)aligned_alloc(64, (int) order * (int) order * sizeof(float));
    image_mat = (unsigned char*)aligned_alloc(64, width * height * sizeof(unsigned char));
    result_mat = (unsigned char*)aligned_alloc(64, width * height * sizeof(unsigned char));

    if (fread(image_mat, sizeof(unsigned char), height * width, input_file) != (size_t)(height * width)) {
        exit(1);
    }
    fclose(input_file);
    
    float sum = 0;
    // for (int i = 0; i < order; i++) {
    //     for (int j = 0; j < order; j++) {
    //         kernel[i * (int) order + j] = (1/(2*M_PI*sigma*sigma)) * 
    //         exp(-(pow(i - floor(order/2), 2) + pow(j - floor(order/2), 2))/(2 * sigma * sigma));
    //         // printf("%.8f ", kernel[i * (int) order + j]);
    //     }
    //     // printf("\n");
    // }

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            kernel[i * (int) order + j] = exp(-(pow(i - floor(order/2), 2) + pow(j - floor(order/2), 2))/(2 * sigma * sigma));
            sum += exp(-(pow(i - floor(order/2), 2) + pow(j - floor(order/2), 2))/(2 * sigma * sigma));
            // printf("%.8f ", kernel[i * (int) order + j]);
        }
        // printf("\n");
    }

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            kernel[i * (int) order + j] /= sum;
            // printf("%.8f ", kernel[i * (int) order + j]);
        }
        // printf("\n");
    }
    
    gaussian_calc(image_mat, result_mat, kernel, width, height, order);

    // /* Save output image */
	write_gaussian(output_filename, result_mat, width, height);
    
	// free(filename);

	return 0;
}