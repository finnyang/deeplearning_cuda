#include "DL/layers/normalize_layer.h"
#include "DL/util/common_function.h"
#include <cblas.h>
#include <malloc.h>
#include <stdlib.h>

void setup_normalize(struct LayerParameter* layer_parameter){
	struct Blob* bottom = layer_parameter->bottoms[0];
	int n = bottom->n;
	int c = bottom->c;
	int h = bottom->h;
	int w = bottom->w;
	struct Blob* learn_param = layer_parameter->learn_parameter[0];
	MakeBlob(n, c, h, w, layer_parameter->tops[0]);
	if(layer_parameter->parameter.normalize_param.channel_shared){
		MakeBlob(1, 1, 1, 1, learn_param);
	}else{
		MakeBlob(1, c, 1, 1, learn_param);
	}
	Doutput_shape_info();
}

__global__ void normalize_across_spatial(float* input, int n, int dim){
	int idx = threadIdx.x;
	float sum = 1e-10;
	int i = 0;
	if(idx < n){
		int temp = n*dim;
		for(i = 0; i < dim; ++i){
			sum+=input[temp+i];
		}
		//sum = pow(sum, float(0.5));
		for(i = 0; i < dim; ++i){
			input[temp+i]/=sum;
			input[temp+i]=pow(input[temp+i], float(0.5));
		}
	}
}
//
//__global__ void normalize_nacross_spatial(float* input, int n, int c, int dim){
//	if(blockIdx.x < n && threadIdx.x < dim){
//		float sum = 1e-10;
//		int i = 0;
//		int temp = dim*blockIdx.x*c+threadIdx.x;
//		for(i = 0; i < c; ++i){
//			sum = sum + input[temp+i*dim];
//		}
//		for(i = 0; i < c; ++i){
//			input[temp+i*dim] = input[temp+i*dim] / sum;
//			input[temp+i*dim]=pow(input[temp+i*dim], float(0.5));
//		}
//	}
//}



__global__ void normalize_nacross_spatial(float* input, int n, int c, int h, int w ){
	if(blockIdx.x < h && threadIdx.x < w){
		float sum = 1e-10;
		int i = 0;
		//int temp = dim*(blockIdx.x*blockDim.x + threadIdx.x);
		int temp = blockIdx.x*blockDim.x+threadIdx.x;
		for(i = 0; i < c; ++i){
			sum= sum + input[temp+i*h*w];
		}
		for(i = 0; i < c; ++i){
			input[temp+i*h*w]= input[temp+i*h*w] / sum;
			input[temp+i*h*w]=pow(input[temp+i*h*w], float(0.5));
		}
	}
}


__global__ void normalize_channel_nshared(float* input, int n, int c, int dim, float* scale){
	if(blockIdx.x < n && threadIdx.x < c)
	{
		int i = 0;
		int temp = dim*(blockIdx.x*blockDim.x + threadIdx.x);
		for(i = 0; i < dim; ++i){
			input[temp+i] = input[temp+i] * scale[threadIdx.x];
		}
	}
}

void forward_normalize(struct LayerParameter* layer_parameter){
	Doutput_info();
	struct Blob* input = layer_parameter->bottoms[0];
	struct Blob* output = layer_parameter->tops[0];
	struct Blob* scale = layer_parameter->learn_parameter[0];
	int n,c,h,w;
	n = input->n;
	c = input->c;
	h = input->h;
	w = input->w;
	power_gpu(input->gpu_data, output->gpu_data, input->count);
//	float* data_input = (float*)malloc(sizeof(float)*input->count);
//	float* data_output = (float*)malloc(sizeof(float)*input->count);
//	cudaMemcpy(data_input, input->gpu_data, sizeof(float)*input->count, cudaMemcpyDeviceToHost);
//	cudaMemcpy(data_output, output->gpu_data, sizeof(float)*input->count, cudaMemcpyDeviceToHost);
//	int j = 0;
//	for(; j < input->count; ++j){
//		printf("%f %f %f\n", data_input[j], data_output[j], data_input[j]*data_input[j]-data_output[j]);
//	}
	int across_spatial = layer_parameter->parameter.normalize_param.across_spatial;
	int channel_shared = layer_parameter->parameter.normalize_param.channel_shared;
	if(across_spatial){
		normalize_across_spatial<<<1, n>>>(output->gpu_data, n, c*h*w);
	}else{
		//dim3 block((h*w+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS,CUDA_NUM_THREADS);
		//normalize_nacross_spatial<<<n, block>>>(output->gpu_data, n, c, h*w);
//		float* sumt;
//		cudaMalloc(&sumt, sizeof(float)*n*h*w);
		int i = 0;
		for(i = 0; i < n; ++i){
			normalize_nacross_spatial<<<h, w>>>(output->gpu_data+i*n*c*h*w, n, c, h, w);
		}
//		float* data = (float*)malloc(sizeof(float)*n*h*w);;
//		cudaMemcpy(data, sumt, sizeof(float)*n*h*w, cudaMemcpyDeviceToHost);
//		int j;
//		for(j = 0; j < n*h*w; ++j){
//			printf("%d %f\n", j, data[j]);
//		}
	}
	if(channel_shared){
		scale_gpu(output->gpu_data, output->gpu_data, output->count, scale->gpu_data[0]);
	}else{
		normalize_channel_nshared<<<n, c>>>(output->gpu_data, n, c, h*w, scale->gpu_data);
		float* data = (float*)malloc(sizeof(float)*scale->count);
		cudaMemcpy(data, scale->gpu_data, sizeof(float)*scale->count, cudaMemcpyDeviceToHost);
//		int j = 0;
//		for(j = 0; j < scale->count; ++j){
//			printf("sacle %d %f\n", j, data[j]);
//		}
		free(data);
	}

//	struct Blob* bottom = layer_parameter->bottoms[0];
//	int n = bottom->n;
//	int c = bottom->c;
//	int h = bottom->h;
//	int w = bottom->w;
//	int dim = c*h*w;
//	int spatial_dim = h*w;
//	int i, j;
//	float* bottom_data = bottom->cpu_data;
//	float* top_data = layer_parameter->tops[0]->cpu_data;
//	float* med = (float*)malloc(sizeof(float)*dim);
//	float* norm;
//	float* norm_ori;
//	float* sum_multiplier = (float*)malloc(sizeof(float)*c);
//	float* sum_spatial_multiplier = (float*)malloc(sizeof(float)*spatial_dim);
//	float* scale = layer_parameter->learn_parameter[0]->cpu_data;
//	int across_spatial = layer_parameter->parameter.normalize_param.across_spatial;
//	int channel_shared = layer_parameter->parameter.normalize_param.channel_shared;
//	for(i = 0; i < c; ++i)
//		sum_multiplier[i] = 1.0;
//	for(i = 0; i < spatial_dim; ++i)
//		sum_spatial_multiplier[i] = 1.0;
//	if(across_spatial){
//		norm = (float*)malloc(sizeof(float)*n);
//		for(i = 0; i < n; ++i)
//			norm[i] = 1e-10;
//	}else{
//		norm = (float*)malloc(sizeof(float)*n*h*w);
//		for(i = n*h*w-1; i >= 0; --i)
//			norm[i] = 1e-10;
//	}
//	norm_ori = norm;
//	for(i = 0; i < n; ++i){
//		for(j = 0; j < dim; ++j){
//			med[j] = bottom_data[j]*bottom_data[j];
//		}
//		if(across_spatial){
//			for(j = 0; j < dim; ++j)
//				norm[i] += med[j];
//			norm[i] = pow(norm[i], 0.5);
//			for(j = 0; j < dim; ++j){
//				top_data[j] = bottom_data[j]/norm[i];
//			}
//		}else{
//			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, spatial_dim, c,
//					1.0, sum_multiplier, c, med, spatial_dim, 0.0, norm, spatial_dim);
//
//			for(j = 0; j < spatial_dim; ++j)
//				norm[j] = pow(norm[j], 0.5);
//			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, spatial_dim, 1,
//					1.0, sum_multiplier, 1, norm, spatial_dim, 0.0, med, spatial_dim);
//			for(j = 0; j < dim; ++j){
//				top_data[j] = bottom_data[j]/med[j];
//			}
//			norm = norm + spatial_dim;
//		}
//		if(channel_shared){
//			for(j = 0; j < dim; ++j)
//				top_data[j] = top_data[j]*scale[0];
//		}else{
//			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, spatial_dim, 1,
//					1.0, scale, 1, sum_spatial_multiplier, spatial_dim, 0.0, med, spatial_dim);
//			for(j = 0; j < dim; ++j){
//				top_data[j] = top_data[j]*med[j];
//			}
//		}
//		bottom_data += dim;
//		top_data += dim;
//	}
//	free(med);
//	free(norm_ori);
//	free(sum_multiplier);
//	free(sum_spatial_multiplier);
}
