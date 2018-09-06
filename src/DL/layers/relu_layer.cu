#include "DL/layers/relu_layer.h"
#include <cuda_runtime.h>

__global__ void gpu_relu(float* input, float* output, int count, float negative_slope){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx < count){
		output[idx] = input[idx] > 0? input[idx]:input[idx]*negative_slope;
	}
}

void setup_relu(struct LayerParameter* layer_parameter){
	layer_parameter->tops[0]->n = layer_parameter->bottoms[0]->n;
	layer_parameter->tops[0]->c = layer_parameter->bottoms[0]->c;
	layer_parameter->tops[0]->h = layer_parameter->bottoms[0]->h;
	layer_parameter->tops[0]->w = layer_parameter->bottoms[0]->w;
	layer_parameter->tops[0]->gpu_data = layer_parameter->bottoms[0]->gpu_data;
	layer_parameter->tops[0]->count = layer_parameter->bottoms[0]->count;
	layer_parameter->tops[0]->own = 0;
	Doutput_shape_info();
}
void forward_relu(struct LayerParameter* layer_parameter){
	Doutput_info();
	int count = layer_parameter->bottoms[0]->count;
	gpu_relu<<<(count+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(layer_parameter->bottoms[0]->gpu_data, layer_parameter->tops[0]->gpu_data,
			count, layer_parameter->parameter.relu_param.negative_slope);
}
