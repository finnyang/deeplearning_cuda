#include "DL/layers/softmax_layer.h"
#include <cblas.h>
#include <malloc.h>
#include <math.h>
#include "DL/util/common_function.h"

void setup_softmax(struct LayerParameter* layer_parameter){
	MakeBlob(layer_parameter->bottoms[0]->n, layer_parameter->bottoms[0]->c, layer_parameter->bottoms[0]->h,
			layer_parameter->bottoms[0]->w, layer_parameter->tops[0]);
	int outer_num, inter_num, i;
	int axis = layer_parameter->parameter.softmax_param.axis;
	int shape[4] = {layer_parameter->bottoms[0]->n, layer_parameter->bottoms[0]->c, layer_parameter->bottoms[0]->h,
				layer_parameter->bottoms[0]->w};
	outer_num = 1;
	inter_num = 1;
	for(i = 0; i < axis; ++i)
		outer_num *= shape[i];
	for(i = axis+1; i < 4; ++i)
		inter_num *= shape[i];
	MakeBlob(inter_num, outer_num, 1, 1, layer_parameter->meds[0]);
	Doutput_shape_info();
}

__global__ void kernel_max(float* input, int inter_num, int outer_num, int channels, float* max_){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int inter_idx = id%inter_num;
	int outer_idx = id/inter_num;
	if(inter_idx < inter_num && outer_idx < outer_num) {
		int idx;
		int i;
		int temp = channels*outer_idx*inter_num;
		idx = outer_idx*inter_num+inter_idx;
		max_[idx] = input[temp];
		for(i = 1; i < channels; ++i){
			if(max_[idx] < input[temp+i]){
				max_[idx] = input[temp+i];
			}
		}
	}
}

__global__ void kernel_minus_max(float* input ,int inter_num, int outer_num, int channels, float* max_){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int inter_idx = id%inter_num;
	int outer_idx = id/inter_num;
	if(inter_idx < inter_num && outer_idx < outer_num) {
		int idx;
		int i;
		int temp = channels*outer_idx*inter_num;
		idx = outer_idx*inter_num+inter_idx;
		for(i = 0; i < channels; ++i){
			input[temp+i]-= max_[idx];
		}
	}
}

__global__ void kernel_sum(float* input ,int inter_num, int outer_num, int channels, float* sum_){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int inter_idx = id%inter_num;
	int outer_idx = id/inter_num;
	if(inter_idx < inter_num && outer_idx < outer_num) {
		int idx;
		int i;
		int temp = channels*outer_idx*inter_num;
		idx = outer_idx*inter_num+inter_idx;
		sum_[idx] = input[temp];
		for(i = 1; i < channels; ++i){
			sum_[idx] += input[temp+i];

		}
	}
}

__global__ void kernel_div(float* input ,int inter_num, int outer_num, int channels, float* sum_){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int inter_idx = id%inter_num;
	int outer_idx = id/inter_num;
	if(inter_idx < inter_num && outer_idx < outer_num) {
		int idx;
		int i;
		int temp = channels*outer_idx*inter_num;
		idx = outer_idx*inter_num+inter_idx;
		for(i = 0; i < channels; ++i){
			input[temp+i] /= sum_[idx];

		}
	}
}

void forward_softmax(struct LayerParameter* layer_parameter){
	Doutput_info();
	int outer_num, inter_num, channels;
	outer_num = layer_parameter->meds[0]->c;
	inter_num = layer_parameter->meds[0]->n;
	channels =  layer_parameter->bottoms[0]->count/(outer_num*inter_num);
	float* bottom_data = layer_parameter->bottoms[0]->gpu_data;
	float* top_data = layer_parameter->tops[0]->gpu_data;

	copy(bottom_data, top_data, layer_parameter->bottoms[0]->count);
//	dim3 block1((inter_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS,(outer_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
//	printf("softmax dimension %d, %d, %d, %d\n", layer_parameter->bottoms[0]->n, layer_parameter->bottoms[0]->c, layer_parameter->bottoms[0]->h, layer_parameter->bottoms[0]->w);
//	printf("inter_num %d outer_num %d\n", inter_num, outer_num);
//	printf("block x %d block y %d\n", (inter_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, (outer_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
////for(;;);
//	kernel_max<<<1, block1>>>(bottom_data, inter_num, outer_num, channels, layer_parameter->meds[0]->gpu_data);
//	kernel_minus_max<<<1, block1>>>(top_data, inter_num, outer_num, channels, layer_parameter->meds[0]->gpu_data);
//	exp_gpu(top_data, top_data, layer_parameter->bottoms[0]->count);
//	kernel_sum<<<1, block1>>>(top_data, inter_num, outer_num, channels, layer_parameter->meds[0]->gpu_data);
//	kernel_div<<<1, block1>>>(top_data, inter_num, outer_num, channels, layer_parameter->meds[0]->gpu_data);
	kernel_max<<<(inter_num*outer_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(bottom_data, inter_num, outer_num, channels, layer_parameter->meds[0]->gpu_data);
	kernel_minus_max<<<(inter_num*outer_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(top_data, inter_num, outer_num, channels, layer_parameter->meds[0]->gpu_data);
	exp_gpu(top_data, top_data, layer_parameter->bottoms[0]->count);
	kernel_sum<<<(inter_num*outer_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(top_data, inter_num, outer_num, channels, layer_parameter->meds[0]->gpu_data);
	kernel_div<<<(inter_num*outer_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(top_data, inter_num, outer_num, channels, layer_parameter->meds[0]->gpu_data);
}
