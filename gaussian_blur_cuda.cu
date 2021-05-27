#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIV_ROUND_UP(n, d)  (((n) + (d) - 1) / (d))
#define cuda_check(ret) _cuda_check((ret), __FILE__, __LINE__)
inline void _cuda_check(cudaError_t ret, const char *file, int line) {
    if (ret != cudaSuccess) {
        fprintf(stderr, "CudaErr: %s (%s:%d)\n", cudaGetErrorString(ret), file, line);
        exit(1);    
    }
}

__constant__ int width_d;
__constant__ int height_d;
__constant__ int order_d;

__global__ void gaussian_calc_kernel(unsigned char *image_mat, unsigned char *result_mat, float *kernel) 
{ //Naive implementation
    float val = 0;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * blockDim.x + tx; //col
    int i = blockIdx.y * blockDim.y + ty; //row

    if (i >= height_d || j >= width_d) {
        return;
    }

    int center = (order_d - 1) / 2;

    for (int x = 0; x < order_d; x++) {
        for (int y = 0; y < order_d; y++) {
            // Min accounts for right and bottom edges
            // Max accounts for left and top edges
            int mat_x = max(0, min(i + x - center, height_d - 1));
            int mat_y = max(0, min(j + y - center, width_d - 1));
            val += image_mat[mat_x * width_d + mat_y] * kernel[x * order_d + y];
        }
    }
    result_mat[i * width_d + j] = (unsigned char) val; 
}


void gaussian_calc(unsigned char *image_mat, unsigned char *result_mat, float *kernel, int width, int height, int order) 
{
    /* Allocate device memory for all matrices */
    float *kernel_d;
    unsigned char *image_mat_d, *result_mat_d;
    cuda_check(cudaMalloc(&kernel_d, order * order * sizeof(float)));
    cuda_check(cudaMalloc(&image_mat_d, width * height * sizeof(unsigned char)));
    cuda_check(cudaMalloc(&result_mat_d, width * height * sizeof(unsigned char)));
    
    /* Copy kernel and image_mat to device */
    cuda_check(cudaMemcpy(kernel_d, kernel, order * order * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(image_mat_d, image_mat, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpyToSymbol(width_d, &width, sizeof(int)));
    cuda_check(cudaMemcpyToSymbol(height_d, &height, sizeof(int)));
    cuda_check(cudaMemcpyToSymbol(order_d, &order, sizeof(int)));

    /* Invoke kernel function */
    dim3 block_dim(32, 32);
    dim3 grid_dim(DIV_ROUND_UP(width, block_dim.x), DIV_ROUND_UP(height, block_dim.y));
    gaussian_calc_kernel<<<grid_dim, block_dim>>>(image_mat_d, result_mat_d, kernel_d);

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
    
    sigma = atof(argv[3]);
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

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            kernel[i * (int) order + j] = exp(-(pow(i - floor(order/2), 2) + pow(j - floor(order/2), 2))/(2 * sigma * sigma));
            sum += kernel[i * (int) order + j];
        }
    }

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            kernel[i * (int) order + j] /= sum;
        }
    }
    
    gaussian_calc(image_mat, result_mat, kernel, width, height,(int) order);

    // /* Save output image */
	write_gaussian(output_filename, result_mat, width, height);
    
    free(kernel);
    free(image_mat);
    free(result_mat);

	return 0;
}