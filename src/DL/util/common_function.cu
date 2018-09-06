#include "DL/util/common_function.h"

__global__ void set_data(float* input, int count, float scale){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx < count)
		input[idx] = scale;
}

void set(float* input, int count, float scale){
	set_data<<<(count+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(input, count, scale);
}

__global__ void copy_data(float* input, float* output, int count){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx < count){
		output[idx] = input[idx];
	}
}

void copy(float* input, float* output, int count){
	copy_data<<<(count+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(input, output, count);
}
__global__ void exp_data(float* input, float* output, int count){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx < count){
		output[idx] = exp(input[idx]);
	}
}

void exp_gpu(float* input, float* output, int count){
	exp_data<<<(count+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(input, output, count);
}

__global__ void power_data(float* input, float* output, int count){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx < count){
		output[idx] = input[idx]*input[idx];
	}
}

void power_gpu(float* input, float* output, int count){
	power_data<<<(count+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(input, output, count);
}

__global__ void scale_data(float* input, float* output, int count, float scale){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx < count){
		output[idx] = input[idx]*scale;
	}
}

void scale_gpu(float* input, float* output, int count, float scale){
	scale_data<<<(count+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(input, output, count, scale);
}
