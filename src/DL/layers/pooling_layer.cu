#include "DL/layers/pooling_layer.h"
#include <float.h>

void setup_pooling(struct LayerParameter* layer_parameter){
	layer_parameter->has_learn_parameter = 0;
	if(layer_parameter->parameter.pooling_param.global_pooling){
		layer_parameter->parameter.pooling_param.kernel_size = layer_parameter->bottoms[0]->h;
	}
	struct PoolingParameter pooling_param = layer_parameter->parameter.pooling_param;
	struct Blob* input = layer_parameter->bottoms[0];
	struct Blob* output = layer_parameter->tops[0];
	output->n = input->n;
	output->c = input->c;
	output->w = ceil((0.0 + input->w + pooling_param.pad*2 - pooling_param.kernel_size)/pooling_param.stride)+1;
	output->h = ceil((0.0 + input->h + pooling_param.pad*2 - pooling_param.kernel_size)/pooling_param.stride)+1;
	MakeBlob(output->n, output->c, output->h, output->w, output);
	Doutput_shape_info();
}

__global__ void pooling_max_kernel(float* input, int kernel_size, int in_n, int in_c, int in_h, int in_w, int pad, int stride,
		float* output, int out_h, int out_w, int count){
	int idx_idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx_idx < count){
		int n,c,h,w;
		int temp1 = (in_c*out_h* out_w);
		int temp2;
		n = idx_idx/temp1;
		temp2 = idx_idx%temp1;
		temp1 = out_h* out_w;
		c = temp2/temp1;
		temp2 = temp2%temp1;
		h = temp2/out_h;
		w = temp2%out_h;
		int speed_up = n*in_c+c;
		int h_ori_temp = h*stride-pad;
		int w_ori_temp = w*stride-pad;
		float max = -FLT_MAX;
		float med;
		int rh, rw;
		int h_ori, w_ori;
		for(rh = 0; rh < kernel_size; ++rh){
			h_ori = h_ori_temp+rh;
			if (h_ori < 0 || h_ori > in_h-1) continue;
			for(rw = 0; rw < kernel_size; ++rw){
				w_ori = w_ori_temp+rw;
				if (w_ori < 0 || w_ori > in_w-1) continue;
				med = input[(speed_up*in_h+h_ori)*in_w+w_ori];
				if (max < med){
					max = med;
				}
			}
		}
		output[idx_idx] = max;
	}
}

__global__ void pooling_max_kernel_new(float* input, int kernel_size, int in_n, int in_c, int in_h, int in_w, int pad, int stride,
		float* output, int out_h, int out_w, int count){
	int n,c,h,w;
	n = blockIdx.x;
	c = blockIdx.y;
	h = threadIdx.x;
	w = threadIdx.y;
	int idx_idx = ((n*in_c+c)*out_h+h)*out_w+w;
	if(n < in_c && c < in_c && h < out_h && w < out_w){
		int speed_up = n*in_c+c;
		int h_ori_temp = h*stride-pad;
		int w_ori_temp = w*stride-pad;
		float max = -FLT_MAX;
		float med;
		int rh, rw;
		int h_ori, w_ori;
		for(rh = 0; rh < kernel_size; ++rh){
			h_ori = h_ori_temp+rh;
			if (h_ori < 0 || h_ori > in_h-1) continue;
			for(rw = 0; rw < kernel_size; ++rw){
				w_ori = w_ori_temp+rw;
				if (w_ori < 0 || w_ori > in_w-1) continue;
				med = input[(speed_up*in_h+h_ori)*in_w+w_ori];
				if (max < med){
					max = med;
				}
			}
		}
		output[idx_idx] = max;
	}
}

__global__ void pooling_ave_kernel_new(float* input, int kernel_size, int in_n, int in_c, int in_h, int in_w, int pad, int stride,
		float* output, int out_h, int out_w, int count){
		int n,c,h,w;
		n = blockIdx.x;
		c = blockIdx.y;
		h = threadIdx.x;
		w = threadIdx.y;
		int idx_idx = ((n*in_c+c)*out_h+h)*out_w+w;
		if(n < in_c && c < in_c && h < out_h && w < out_w){
				int speed_up = n*in_c+c;
				int h_ori_temp = h*stride-pad;
				int w_ori_temp = w*stride-pad;
				int rh, rw;
				int h_ori, w_ori;
				float sum = 0.0;
				for(rh = 0; rh < kernel_size; ++rh){
					h_ori = h_ori_temp+rh;
					if (h_ori < 0 || h_ori > in_h-1) continue;
					for(rw = 0; rw < kernel_size; ++rw){
						w_ori = w_ori_temp+rw;
						if (w_ori < 0 || w_ori > in_w-1) continue;
						sum += input[(speed_up*in_h+h_ori)*in_w+w_ori];
					}
				}
				output[idx_idx] = sum/(kernel_size*kernel_size);
			}
}

__global__ void pooling_ave_kernel(float* input, int kernel_size, int in_n, int in_c, int in_h, int in_w, int pad, int stride,
		float* output, int out_h, int out_w, int count){
	int idx_idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx_idx < count){
		int n,c,h,w;
		int temp1 = (in_c*out_h* out_w);
		int temp2;
		n = idx_idx/temp1;
		temp2 = idx_idx%temp1;
		temp1 = out_h* out_w;
		c = temp2/temp1;
		temp2 = temp2%temp1;
		h = temp2/out_h;
		w = temp2%out_h;

		int speed_up = n*in_c+c;
		int h_ori_temp = h*stride-pad;
		int w_ori_temp = w*stride-pad;
		float med;
		int rh, rw;
		int h_ori, w_ori;
		float sum = 0.0;
		for(rh = 0; rh < kernel_size; ++rh){
			h_ori = h_ori_temp+rh;
			if (h_ori < 0 || h_ori > in_h-1) continue;
			for(rw = 0; rw < kernel_size; ++rw){
				w_ori = w_ori_temp+rw;
				if (w_ori < 0 || w_ori > in_w-1) continue;
				med = input[(speed_up*in_h+h_ori)*in_w+w_ori];
				sum += med;
			}
		}
		output[idx_idx] = sum/(kernel_size*kernel_size);
	}
}

void forward_pooling(struct LayerParameter* layer_parameter){
	Doutput_info();
	struct Blob* input = layer_parameter->bottoms[0];
	struct Blob* output = layer_parameter->tops[0];
	struct PoolingParameter pooling_param = layer_parameter->parameter.pooling_param;
	int pad = pooling_param.pad;
	int stride = pooling_param.stride;
	int kernel_size = pooling_param.kernel_size;
	int out_h = output->h;
	int out_w = output->w;
	int in_h = input->h;
	int in_w = input->w;
	int in_c = input->c;
	int in_n = input->n;
	if(layer_parameter->parameter.pooling_param.pooling_type == MAX){
		pooling_max_kernel<<<(output->count+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(input->gpu_data, kernel_size, in_n, in_c, in_h, in_w, pad, stride,
				output->gpu_data, out_h, out_w, output->count);
//		dim3  grid((in_n+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS,(in_c+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
//		dim3 block((out_h+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS,(out_w+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
//		pooling_max_kernel_new<<<grid, block>>>(input->gpu_data, kernel_size, in_n, in_c, in_h, in_w, pad, stride,
//						output->gpu_data, out_h, out_w, output->count);
	}else{
		if(layer_parameter->parameter.pooling_param.pooling_type == AVE){
			pooling_ave_kernel<<<(output->count+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(input->gpu_data, kernel_size, in_n, in_c, in_h, in_w, pad, stride,
					output->gpu_data, out_h, out_w, output->count);
//			dim3  grid((in_n+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS,(in_c+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
//			dim3 block((out_h+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS,(out_w+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
//			pooling_ave_kernel_new<<<grid, block>>>(input->gpu_data, kernel_size, in_n, in_c, in_h, in_w, pad, stride,
//									output->gpu_data, out_h, out_w, output->count);
		}
	}
}
