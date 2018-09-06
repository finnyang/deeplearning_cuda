#include "DL/layers/concat_layer.h"

void setup_concat(struct LayerParameter* layer_parameter){
	int bottom_size = layer_parameter->bottom_num;
	int shape[4] = {layer_parameter->bottoms[0]->n, layer_parameter->bottoms[0]->c, layer_parameter->bottoms[0]->h, layer_parameter->bottoms[0]->w};
	int axis = layer_parameter->parameter.concat_param.axis;
	int i;
	if(axis == 0){
		for (i = 1; i < bottom_size; ++i){
			shape[0] += layer_parameter->bottoms[i]->n;
		}
	}else{
		if(axis == 1){
			for (i = 1; i < bottom_size; ++i){
				shape[1] += layer_parameter->bottoms[i]->c;
			}
		}else{
			if(axis == 2){
				for (i = 1; i < bottom_size; ++i){
					shape[2] += layer_parameter->bottoms[i]->h;
				}
			}else{
				if(axis == 3){
					for (i = 1; i < bottom_size; ++i){
						shape[3] += layer_parameter->bottoms[i]->w;
					}
				}
			}
		}
	}
	MakeBlob(shape[0], shape[1], shape[2], shape[3], layer_parameter->tops[0]);
	Doutput_shape_info();
}
//input 的维度是outer_num*inter_num　根据这两个维度以及线程的维度就可以确定输入的维度分别为(blockIdx.x,threadIdx.x)
//output 的维度是outer_num*sum_inter_num，
//现在确定将input放在哪个位置位置为(blockIdx.x,threadIdx.x+n_sum_inter_num)
__global__ void concat_kernel(float* input, float* output,
		int outer_num, int inter_num, int sum_inter_num, int n_sum_inter_num){
	int id = blockDim.x*blockIdx.x+threadIdx.x;
	//int idx = blockIdx.x;
	//int idy = threadIdx.x;
	int idx = id/inter_num;
	int idy = id%inter_num;
	if(idx < outer_num && idy < inter_num){
		//int input_idx = idx*inter_num+idy;
		int output_idx = idx*sum_inter_num+idy+n_sum_inter_num;
		output[output_idx] = input[id];
	}
}

void forward_concat(struct LayerParameter* layer_parameter){
	Doutput_info();
//	int axis = layer_parameter->parameter.concat_param.axis;
//	int shape[4] = {layer_parameter->tops[0]->n, layer_parameter->tops[0]->c, layer_parameter->tops[0]->h, layer_parameter->tops[0]->w};
//	int outer_num=1, inter_num=1;
//	int i,j,k;
//	int bottom_size = layer_parameter->bottom_num;
//	float* top_data = layer_parameter->tops[0]->cpu_data;
//	int temp, temp1;
//	float* bottom_data;
//	for(i = 0; i < axis; ++i)
//		outer_num *= shape[i];
//	for(i = axis+1; i < 4; ++i)
//		inter_num *= shape[i];
//	for(i = 0; i < outer_num; ++i){
//		for(j = 0; j < bottom_size; ++j){
//			if(axis == 0){
//				temp = layer_parameter->bottoms[j]->n*i*inter_num;
//				temp1 = layer_parameter->bottoms[j]->n*inter_num;
//			}else{
//				if(axis == 1){
//					temp = layer_parameter->bottoms[j]->c*i*inter_num;
//					temp1 = layer_parameter->bottoms[j]->c*inter_num;
//				}else{
//					if(axis == 2){
//						temp = layer_parameter->bottoms[j]->h*i*inter_num;
//						temp1 = layer_parameter->bottoms[j]->h*inter_num;
//					}else{
//						if(axis ==3){
//							temp = layer_parameter->bottoms[j]->w*i*inter_num;
//							temp1 = layer_parameter->bottoms[j]->w*inter_num;
//						}
//					}
//				}
//			}
//			bottom_data = layer_parameter->bottoms[j]->cpu_data+temp;
//			for(k = 0; k < temp1; ++k){
//				top_data[0] = bottom_data[k];
//				++top_data;
//			}
//		}
//	}
	int i;
	int bottom_size = layer_parameter->bottom_num;
	int axis = layer_parameter->parameter.concat_param.axis;
	int shape[4] = {layer_parameter->tops[0]->n, layer_parameter->tops[0]->c, layer_parameter->tops[0]->h, layer_parameter->tops[0]->w};
	int bottom_shape[4];
	int outer_num=1;
	int inter_num=1;
	for(i = 0; i < axis; ++i)
		outer_num *= shape[i];
	for(i = axis+1; i < 4; ++i)
		inter_num *= shape[i];
	int n_sum_inter_num=0;
	int sum_inter_num = shape[axis]*inter_num;
	for(i = 0; i < bottom_size; ++i){
		bottom_shape[0] = layer_parameter->bottoms[i]->n;
		bottom_shape[1] = layer_parameter->bottoms[i]->c;
		bottom_shape[2] = layer_parameter->bottoms[i]->h;
		bottom_shape[3] = layer_parameter->bottoms[i]->w;
		inter_num*=bottom_shape[axis];
		//concat_kernel<<<(outer_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, (inter_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS>>>(layer_parameter->bottoms[i]->gpu_data,
		//		layer_parameter->tops[0]->gpu_data, outer_num, inter_num, sum_inter_num, n_sum_inter_num);
		concat_kernel<<<(outer_num*inter_num+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(layer_parameter->bottoms[i]->gpu_data,
						layer_parameter->tops[0]->gpu_data, outer_num, inter_num, sum_inter_num, n_sum_inter_num);
		n_sum_inter_num+=inter_num;
		inter_num/=bottom_shape[axis];
//		cudaError_t cudaerror = cudaPeekAtLastError();
//		if(cudaerror == cudaSuccess){
//			printf("success\n");
//		}else{
//			printf("%s \n", cudaGetErrorString(cudaerror));
//		}
	}
//		cudaError_t cudaerror = cudaPeekAtLastError();
//		if(cudaerror == cudaSuccess){
//			printf("success\n");
//		}else{
//			printf("%s \n", cudaGetErrorString(cudaerror));
//		}
//		for(;;);
}
