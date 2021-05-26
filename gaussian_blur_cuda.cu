#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TILE_WIDTH 32
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
            int mat_x = max(0, min(height_d - 1, i + x - center));
            int mat_y = max(0, min(width_d - 1, j + y - center));

            val += image_mat[mat_x * height_d + mat_y] * kernel[x * order_d + y];
            // if (x <= center - i && y <= center - j) {
            //     val += (image_mat[0] * kernel[x * order_d + y]);
            // } else if (x <= center - i && j + y >= width_d + center - 1) {
            //     val += (image_mat[width_d - 1] * kernel[x * order_d + y]);
            // } else if (i + x >= height_d + center - 1 && y <= center - j) {
            //     val += (image_mat[height_d * (height_d - 1)] * kernel[x * order_d + y]);
            // } else if (i + x >= height_d + center - 1 && j + y >= width_d + center - 1) {
            //     val += (image_mat[height_d * (height_d - 1) + width_d - 1] * kernel[x * order_d + y]);
            // } else if (x < center - i) {
            //     val += (image_mat[j + (y - center)] * kernel[x * order_d + y]);
            // } else if (y < center - j) {
            //     val += (image_mat[(i + (x - center)) * height_d] * kernel[x * order_d + y]);
            // } else if (i + x > height_d + center - 1) {
            //     val += (image_mat[height_d * (height_d - 1) + j + (y - center)] * kernel[x * order_d + y]);
            // } else if (j + y > width_d + center - 1) {
            //     val += (image_mat[(i + (x - center)) * height_d + width_d - 1] * kernel[x * order_d + y]);
            // } else {
            //     val += (image_mat[(i + (x - center)) * height_d + j + (y - center)] * kernel[x * order_d + y]);
            // }
        }
    }
    result_mat[i * height_d + j] = (unsigned char) val; 
    val = 0;
}


// __global__ void gaussian_calc_kernel(unsigned char *image_mat, unsigned char *result_mat, float *kernel, int width, int height, float order) 
// { // Shared memory implementation
//     __shared__ unsigned char image_s[TILE_WIDTH][TILE_WIDTH];
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int j = blockIdx.x * TILE_WIDTH + tx; //col
//     int i = blockIdx.y * TILE_WIDTH + ty; //row

//     if (j >= height || i >= width) {
//         return;
//     }
//     image_s[ty][tx] = image_mat[i * width + j];
//     __syncthreads();
//     float val = 0;
    

//     int center = ((int) order - 1) / 2;

//     for (int x = 0; x < (int) order; x++) {
//         for (int y = 0; y < (int) order; y++) {
//             if (x <= center - i && y <= center - j) {
//                 val += (image_s[0][0] * kernel[x * (int) order + y]);
//             } else if (x <= center - i && j + y >= width + center - 1) {
//                 val += (image_s[0][TILE_WIDTH - 1] * kernel[x * (int) order + y]);
//             } else if (i + x >= height + center - 1 && y <= center - j) {
//                 val += (image_s[TILE_WIDTH - 1][0] * kernel[x * (int) order + y]);
//             } else if (i + x >= height + center - 1 && j + y >= width + center - 1) {
//                 val += (image_s[TILE_WIDTH - 1][TILE_WIDTH - 1] * kernel[x * (int) order + y]);
//             } else if (x < center - i) { //row too high
//                 val += (image_s[0][j + (y - center)] * kernel[x * (int) order + y]);
//             } else if (y < center - j) { // col too far left
//                 val += (image_s[(i + (x - center)) * TILE_WIDTH][0] * kernel[x * (int) order + y]);
//             } else if (i + x > height + center - 1) { // row too low
//                 val += (image_s[TILE_WIDTH - 1][j + (y - center)] * kernel[x * (int) order + y]);
//             } else if (j + y > width + center - 1) { // col too far right
//                 val += (image_s[(i + (x - center)) * height + width - 1][TILE_WIDTH - 1] * kernel[x * (int) order + y]);
//             } else {
//                 val += (image_s[i][j] * kernel[x * (int) order + y]);
//             }
//         }
//     }
//     result_mat[i * height + j] = (unsigned char) val; 
//     val = 0;
// }

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
    dim3 grid_dim(DIV_ROUND_UP(height, block_dim.x), DIV_ROUND_UP(width, block_dim.y));
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
    
    gaussian_calc(image_mat, result_mat, kernel, width, height,(int) order);

    // /* Save output image */
	write_gaussian(output_filename, result_mat, width, height);
    
	// free(filename);

	return 0;
}