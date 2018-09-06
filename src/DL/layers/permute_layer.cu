#include "DL/layers/permute_layer.h"
#include "DL/util/common_function.h"

void setup_permute(struct LayerParameter* layer_parameter){
	int idx_n = layer_parameter->parameter.permute_param.idx_n;
	int idx_c = layer_parameter->parameter.permute_param.idx_c;
	int idx_h = layer_parameter->parameter.permute_param.idx_h;
	int idx_w = layer_parameter->parameter.permute_param.idx_w;
	int shape[4] = {layer_parameter->bottoms[0]->n, layer_parameter->bottoms[0]->c, layer_parameter->bottoms[0]->h, layer_parameter->bottoms[0]->w};
	int n = shape[idx_n];
	int c = shape[idx_c];
	int h = shape[idx_h];
	int w = shape[idx_w];
	MakeBlob(n, c, h, w, layer_parameter->tops[0]);
	Doutput_shape_info();
}

__global__ void permute_kernel(float* input, int in_n, int in_c, int in_h, int in_w,
		float* output, int out_n, int out_c, int out_h, int out_w,
		int idx_n, int idx_c, int idx_h, int idx_w){
	//int idx[4] = { blockIdx.y, blockIdx.x, threadIdx.x, threadIdx.y};
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int i_n, i_c, i_h, i_w;
	int temp = in_c*in_h*in_w;
	i_n = index/temp;
	temp = index%temp;
	i_c = temp/(in_w*in_h);
	temp = temp%(in_w*in_h);
	i_h = temp/in_w;
	i_w = temp%in_w;
	//int idx[4] = {blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y};
	int idx[4] = {i_n, i_c, i_h, i_w};
	if(idx[0] < in_n && idx[1] < in_c && idx[2] < in_h && idx[3] < in_w){

		int idx_input = ((idx[0]*in_c+idx[1])*in_h+idx[2])*in_w+idx[3];

		int idx_output = ((idx[idx_n]*out_c+idx[idx_c])*out_h+idx[idx_h])*out_w+idx[idx_w];
		output[idx_output] = input[idx_input];
	}
}

void forward_permute(struct LayerParameter* layer_parameter){
	Doutput_info();
//	int idx_n = layer_parameter->parameter.permute_param.idx_n;
//	int idx_c = layer_parameter->parameter.permute_param.idx_c;
//	int idx_h = layer_parameter->parameter.permute_param.idx_h;
//	int idx_w = layer_parameter->parameter.permute_param.idx_w;
//	int shape_bottom[4] = {layer_parameter->bottoms[0]->n, layer_parameter->bottoms[0]->c,
//			layer_parameter->bottoms[0]->h, layer_parameter->bottoms[0]->w};
//	int shape_top[4] = {shape_bottom[idx_n], shape_bottom[idx_c], shape_bottom[idx_h], shape_bottom[idx_w]};
//	float* top_data = layer_parameter->tops[0]->cpu_data;
//	float* bottom_data = layer_parameter->bottoms[0]->cpu_data;
//	int n,c,h,w;
//	int top_idx,bottom_idx;
//	int b_flex[4];
//	int f_flex[4] = {idx_n, idx_c, idx_h, idx_w};
//	for(n = 0; n < shape_top[0]; ++n){
//		for(c = 0; c < shape_top[1]; ++c){
//			for(h = 0; h < shape_top[2]; ++h){
//				for(w = 0; w < shape_top[3]; ++w){
//					b_flex[f_flex[0]] = n;
//					b_flex[f_flex[1]] = c;
//					b_flex[f_flex[2]] = h;
//					b_flex[f_flex[3]] = w;
//					top_idx = ((n*shape_top[1]+c)*shape_top[2]+h)*shape_top[3]+w;
//					bottom_idx = ((b_flex[0]*shape_bottom[1]+b_flex[1])*shape_bottom[2]+b_flex[2])*shape_bottom[3]+b_flex[3];
//					top_data[top_idx] = bottom_data[bottom_idx];
//				}
//			}
//		}
//	}
	int in_n, in_c, in_h, in_w;
	int out_n, out_c, out_h, out_w;
	in_n = layer_parameter->bottoms[0]->n;
	in_c = layer_parameter->bottoms[0]->c;
	in_h = layer_parameter->bottoms[0]->h;
	in_w = layer_parameter->bottoms[0]->w;
	out_n = layer_parameter->tops[0]->n;
	out_c = layer_parameter->tops[0]->c;
	out_h = layer_parameter->tops[0]->h;
	out_w = layer_parameter->tops[0]->w;
	//dim3 grid((in_n+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, (in_c+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
	//dim3 block((in_h+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, (in_w+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
//	dim3 grid(2147483647, 65535, 65535);
//	dim3 block(10224);
//	printf("permute dimension %d %d %d %d\n", in_n, in_c, in_h, in_w);
//	printf("permute dimension %d %d %d %d\n", layer_parameter->parameter.permute_param.idx_n, layer_parameter->parameter.permute_param.idx_c,
//			layer_parameter->parameter.permute_param.idx_h, layer_parameter->parameter.permute_param.idx_w);
	//for(;;);
	//copy(layer_parameter->bottoms[0]->gpu_data, layer_parameter->tops[0]->gpu_data, layer_parameter->tops[0]->count);
	permute_kernel<<<(layer_parameter->bottoms[0]->count+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(layer_parameter->bottoms[0]->gpu_data, in_n, in_c, in_h, in_w,
			layer_parameter->tops[0]->gpu_data, out_n, out_c, out_h, out_w,
			layer_parameter->parameter.permute_param.idx_n, layer_parameter->parameter.permute_param.idx_c,
			layer_parameter->parameter.permute_param.idx_h, layer_parameter->parameter.permute_param.idx_w);
//	permute_kernel<<<1, block>>>(layer_parameter->bottoms[0]->gpu_data, in_n, in_c, in_h, in_w,
//				layer_parameter->tops[0]->gpu_data, out_n, out_c, out_h, out_w,
//				layer_parameter->parameter.permute_param.idx_n, layer_parameter->parameter.permute_param.idx_c,
//				layer_parameter->parameter.permute_param.idx_h, layer_parameter->parameter.permute_param.idx_w);
//	cudaError_t cudaerror = cudaPeekAtLastError();
//	if(cudaerror == cudaSuccess){
//		printf("success\n");
//	}else{
//		printf("%s \n", cudaGetErrorString(cudaerror));
//	}
//	for(;;);
}
