#pragma optimize("", off);

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

// Example data to load from your file:
// 117,85,146,194,21,20,20,20,20,20,20,20,20,20,20,20,20,
// 20,20,20,20,20,20,20,21,22,417,418,141,68,196,198,194,177,
// 173,173,172,2101,172,172,173,149,172,172,172,173,172,175,
// 173,173,172,171,172,100,111,101,101,100,98,98,98,88,
// 98,99,97,98,96,96,97,98,98,96,98,98,97,98,97,97,92,96


// A few filtering kernels as samples
float low_pass_kernel[9] = {
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0
};

// float the inputs
float hor_line_kernel[9] = {
    -1.0, -1.0, -1.0,
     2.0,  2.0,  2.0,
    -1.0, -1.0, -1.0
};

float ver_line_kernel[9] = {
    -1.0, 2.0, -1.0,
    -1.0, 2.0, -1.0,
    -1.0, 2.0, -1.0
};


__global__ void gpu_calculation_loop(int* distance_matrix, float* filtered_matrix, int dstNum, int posNum) {
    int i = threadIdx.x;
    // Bound threads count to the length of array
    // In a real CUDA program usually more threads will be executed than there are elements in the array

    float low_pass_kernel[9] = {
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0
    };

    // float the inputs
    float hor_line_kernel[9] = {
        -1.0, -1.0, -1.0,
         2.0,  2.0,  2.0,
        -1.0, -1.0, -1.0
    };

    float ver_line_kernel[9] = {
        -1.0, 2.0, -1.0,
        -1.0, 2.0, -1.0,
        -1.0, 2.0, -1.0
    };

    int l, j, k, x, y;
    float sum = 0.0;

    // Repeat 1000 times
    for (l = 0; l < 1000; l++) {
        printf("%d\n", l);
        /*
        // Apply kernel for all points in the matrix
        for (y = 1; y < dstNum - 1; y++) {
            for (x = 1; x < posNum - 1; x++) {
                sum = 0.0;
                for (k = -1; k < 2; k++) {
                    for (j = -1; j < 2; j++) {
                        sum += hor_line_kernel[(k + 1) * 3 + (j + 1)] * (float)distance_matrix[(y - k) * posNum + (x - j)];
                        printf("y[%d] x[%d] k[%d] j[%d] | kernel[%d]: %d | matrix[%d]: %f\n", y, x, k, j, (k + 1) * 3 + (j + 1), hor_line_kernel[(k + 1) * 3 + (j + 1)], (y - k) * posNum + (x - j), distance_matrix[(y - k) * posNum + (x - j)]);
                    }
                }
                filtered_matrix[y * posNum + x] = sum / 255;
            }
        }*/
    }
}

void gpu_calculation(int* distance_matrix, float* filtered_matrix, int dstNum, int posNum) {
    // Host memory array
    // Already allocated
    // GPU memory array
    int* device_distance_matrix; cudaMalloc((void**)&distance_matrix, posNum * sizeof(int));
    float* device_filtered_matrix; cudaMalloc((void**)&distance_matrix, posNum * sizeof(float));
    // Copy Host memory to Device
    cudaMemcpy(&device_distance_matrix, distance_matrix, posNum * sizeof(int), cudaMemcpyHostToDevice);
    // Block and Grid dimensions
    dim3 grid_size(1); dim3 block_size(posNum);
    // Launch GPU kernel
    gpu_calculation_loop << <grid_size, block_size >> > (distance_matrix, filtered_matrix, dstNum, posNum);
    // Copy results back to Host memory
    cudaMemcpy(filtered_matrix, &device_filtered_matrix, posNum * sizeof(int), cudaMemcpyDeviceToHost);
    // Free memory allocated
    cudaFree(device_filtered_matrix);
    cudaFree(device_distance_matrix);
    // for (int i = 0; i < size_N; i++) { printf("Array GPU[%d]: %d\n", i, host_array[i]);}
}


int main3(int argc, char* argv[]) {

    clock_t start, end;
    double cpu_time_used;

    if (argc < 2) {
        printf("Need 2 arguments! X(Number of positions) and Y(Max Distance)\n\n");
        return -1;
    }

    
    int posNum = atoi(argv[1]);
    int dstNum = atoi(argv[2]);
    printf("Positions: %d, Max Distance: %d\n", posNum, dstNum);

    int* distance_vector = (int*)calloc(posNum, sizeof(int));
    int* distance_matrix = (int*)calloc(posNum * dstNum, sizeof(int));
    float* filtered_matrix = (float*)calloc(posNum * dstNum, sizeof(float));
    int* threshold_matrix = (int*)calloc(posNum * dstNum, sizeof(int));
    int* new_vector = (int*)calloc(posNum, sizeof(int));

    int i;


    // Implement your LOAD_DATA function here to load X number of elements and store them into distance_vector
    int data[] = { 117,85,146,194,21,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,21,22,417,418,141,68,196,198,194,177,173,173,172,2101,172,172,173,149,172,172,172,173,172,175,173,173,172,171,172,100,111,101,101,100,98,98,98,88,98,99,97,98,96,96,97,98,98,96,98,98,97,98,97,97,92,96 };
    for (int i = 0; i < posNum; i++){
        distance_vector[i] = data[i];
    }

    // Creates matrix from input vector
    for (i = 0; i < posNum; i++) {
        int distance = distance_vector[i];
        if (distance >= dstNum) distance = dstNum - 1;
        distance_matrix[distance * posNum + i] = 255;//sets distance object
        printf("distance_matrix[%d]: %d\n", distance * posNum + i, distance_matrix[distance * posNum + i]);
    }

    // Start time measure
    start = clock();

    /******************* OPTIMIZE THIS ***********************/
    int l, j, k, x, y;
    float sum = 0.0;
    gpu_calculation(distance_matrix, filtered_matrix, dstNum, posNum);
    /********************************************************/

        // End time measure
    end = clock();
    cpu_time_used = ((double)(end - start) / 1000) / CLOCKS_PER_SEC;

    // Threshold the matrix
    for (x = 0; x < posNum; x++) {
        for (y = 0; y < dstNum; y++) {
            if (filtered_matrix[y * posNum + x] >= 4.0) {
                threshold_matrix[y * posNum + x] = 1;
            }
        }
    }

    // Extract vector from matrix
    for (x = 0; x < posNum; x++) {
        for (y = 0; y < dstNum; y++) {
            if (threshold_matrix[y * posNum + x]) {
                new_vector[x] = y;//sets distance object
            }
        }
        if (new_vector[x] == 0) new_vector[x] = 300;
    }

    // Prints threshold-filtered vector
    for (x = 0; x < posNum; x++) {
        printf("%d, ", new_vector[x]);
    }

    printf("\n Total time = %f ms\n", cpu_time_used * 1000);

    free(distance_vector);
    free(distance_matrix);
    free(filtered_matrix);
    free(threshold_matrix);
    free(new_vector);

    return 0;
}