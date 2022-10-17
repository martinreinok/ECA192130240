#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>



void increment_cpu(int size_N) {
	int* array = (int*)calloc(size_N, sizeof(int));
	for (int i = 0; i < size_N; i++) {
		array[i] = array[i] + pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, i + i * i))))))))))))))))))))))))))))))));
		// printf("Array CPU[%d]: %d\n", i, array[i]);
	}
	free(array);
}


__global__ void increment_gpu_kernel(int *device_array, int size_N) {
	int i = threadIdx.x;
	// Bound threads count to the length of array
	// In a real CUDA program usually more threads will be executed than there are elements in the array
	if (i < size_N) {
		device_array[i] = device_array[i] + pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, pow(i * i, i + i * i))))))))))))))))))))))))))))))));
	}
}

void increment_gpu(int size_N) {
	// Host memory array
	int* host_array = (int*)calloc(size_N, sizeof(int));
	// GPU memory array
	int* device_array; cudaMalloc((void**)&device_array, size_N * sizeof(int));
	// Copy Host memory to Device
	cudaMemcpy(device_array, host_array, size_N*sizeof(int), cudaMemcpyHostToDevice);
	// Block and Grid dimensions
	dim3 grid_size(1); dim3 block_size(size_N);
	// Launch GPU kernel
	increment_gpu_kernel<<<grid_size, block_size>>>(device_array, size_N);
	// Copy results back to Host memory
	cudaMemcpy(host_array, device_array, size_N * sizeof(int), cudaMemcpyDeviceToHost);
	// Free memory allocated
	cudaFree(device_array);
	// for (int i = 0; i < size_N; i++) { printf("Array GPU[%d]: %d\n", i, host_array[i]);}
}

int main() {
	// BreakEven~:   1000000
	int array_size = 10000000;
	// CPU
	auto start = std::chrono::high_resolution_clock::now();
	increment_cpu(array_size);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	printf("\n Total time CPU = %d ms\n", duration.count());
	// GPU
	start = std::chrono::high_resolution_clock::now();
	increment_gpu(array_size);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	printf("\n Total time GPU = %d ms\n", duration.count());
}