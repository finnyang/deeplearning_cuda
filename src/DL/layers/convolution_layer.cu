#include "DL/layers/convolution_layer.h"
#include "DL/util/im2col.h"
#include "DL/util/common_function.h"
#include <cblas.h>
#include <malloc.h>

void setup_convolution(struct LayerParameter* layer_parameter){
	struct ConvolutionParameter convolution_param = layer_parameter->parameter.convolution_param;
	struct Blob* input = layer_parameter->bottoms[0];
	struct Blob* output = layer_parameter->tops[0];
	struct Blob* weights = layer_parameter->learn_parameter[0];
	struct Blob* bias = layer_parameter->learn_parameter[1];
	output->n = input->n;
	output->c = convolution_param.num_output;
	output->w = (input->w + convolution_param.pad*2 - (convolution_param.dilation*(convolution_param.kernel_size-1)+1))/convolution_param.stride+1;
	output->h = (input->h + convolution_param.pad*2 - (convolution_param.dilation*(convolution_param.kernel_size-1)+1))/convolution_param.stride+1;
	MakeBlob(output->n, output->c, output->h, output->w, output);
	MakeBlob(convolution_param.num_output, input->c, convolution_param.kernel_size, convolution_param.kernel_size, weights);
	MakeBlob(convolution_param.num_output, 1, 1, 1, bias);
	if(convolution_param.pad == 0 && convolution_param.dilation == 1 && convolution_param.stride == 1){
		MakeBlob(output->w, output->h, 1, 1, layer_parameter->meds[0]);
	}else{
		MakeBlob(output->w, output->h, 1, 1, layer_parameter->meds[0]);
		MakeBlob(convolution_param.kernel_size*convolution_param.kernel_size*input->c, output->w*output->h, 1, 1, layer_parameter->meds[1]);
	}
	set(layer_parameter->meds[0]->gpu_data, layer_parameter->meds[0]->count, 1.0);
	Doutput_shape_info();
}
void forward_convolution(struct LayerParameter* layer_parameter){
	Doutput_info();
	struct Blob* input = layer_parameter->bottoms[0];
	struct Blob* output = layer_parameter->tops[0];
	struct Blob* weights = layer_parameter->learn_parameter[0];
	struct Blob* bias = layer_parameter->learn_parameter[1];
	int batch   = input->n;
	int size = input->count/batch;
	int input_w = input->w;
	int input_h = input->h;
	int input_c = input->c;
	int stride = layer_parameter->parameter.convolution_param.stride;
	int pad = layer_parameter->parameter.convolution_param.pad;
	int dilation = layer_parameter->parameter.convolution_param.dilation;
	int output_w = output->w;
	int output_h = output->h;
	int kernel_c = layer_parameter->learn_parameter[0]->c;
	int kernel_size = layer_parameter->learn_parameter[0]->h;
	int out_size = output->count/batch;
	int i= 0;
	int M=layer_parameter->parameter.convolution_param.num_output;
	int N=output_w*output_h;
	int K=kernel_size*kernel_size*kernel_c;
//	float* multi ;
//	cudaMalloc(&(multi), sizeof(float)*N);
//	//set_data<<<(N+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>> (multi, N, 1.0);
//	set(multi, N, 1.0);
	if(pad == 0 && dilation == 1 && stride == 1){;
		const float a = 1.0;
		const float b = 0.0;
		for(i = 0; i < batch; ++i){
			cublasSgemm(*(layer_parameter->p_cublas_handle), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
					&a, input->gpu_data+i*size, N, weights->gpu_data, K, &b, output->gpu_data+i*out_size, N);
			cublasSgemm(*(layer_parameter->p_cublas_handle), CUBLAS_OP_N, CUBLAS_OP_N, N, M, 1,
					&a, layer_parameter->meds[0]->gpu_data, N, bias->gpu_data, 1, &a, output->gpu_data+i*out_size, N);
		}
	}else{
		struct Blob* med = (struct Blob*)malloc(sizeof(struct Blob));
		MakeBlob(1, 1, K, N, med);
		const float a = 1.0;
		const float b = 0.0;
		for(i = 0; i < batch; ++i){
			im2col_gpu(input->gpu_data+i*size, input_c, input_h, input_w, kernel_size, stride, pad, dilation, layer_parameter->meds[1]->gpu_data);
			/*float* data = (float*)malloc(sizeof(float)*K*N);
			cudaMemcpy(data, med->gpu_data, sizeof(float)*K*N, cudaMemcpyDeviceToHost);
			FILE* fp = fopen("/home/yang/Documents/dl_back_now_change/deeplearning/med.txt", "r");
			float* data1 = (float*)malloc(sizeof(float)*K*N);
			fread(data1, sizeof(float),K*N, fp);
			for(j = 0; j < K*N; ++j){
				if(abs(data[j]-data1[j]) > 0.00001){
					printf("med %d %f %f\n", j, data[j], data1[j]);
				}
			}
			free(data1);
			free(data);


			data = (float*)malloc(sizeof(float)*weights->count);
			cudaMemcpy(data, weights->gpu_data, sizeof(float)*weights->count, cudaMemcpyDeviceToHost);
			fp = fopen("/home/yang/Documents/dl_back_now_change/deeplearning/weight.txt", "r");
			data1 = (float*)malloc(sizeof(float)*weights->count);
			fread(data1, sizeof(float),weights->count, fp);
			for(j = 0; j < weights->count; ++j){
				if(abs(data[j]-data1[j]) > 0.00001){
					printf("weights %d %f %f\n", j, data[j], data1[j]);
				}
			}
			free(data1);
			free(data);

			data = (float*)malloc(sizeof(float)*bias->count);
			cudaMemcpy(data, bias->gpu_data, sizeof(float)*bias->count, cudaMemcpyDeviceToHost);
			fp = fopen("/home/yang/Documents/dl_back_now_change/deeplearning/bias.txt", "r");
			data1 = (float*)malloc(sizeof(float)*bias->count);
			fread(data1, sizeof(float),bias->count, fp);
			for(j = 0; j < bias->count; ++j){
				if(abs(data[j]-data1[j]) > 0.00001){
					printf("bias %d %f %f\n", j, data[j], data1[j]);
				}
			}
			free(data1);
			free(data);
*/
			cublasSgemm(*(layer_parameter->p_cublas_handle), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
					&a, layer_parameter->meds[1]->gpu_data, N, weights->gpu_data, K, &b, output->gpu_data+i*out_size, N);
			cublasSgemm(*(layer_parameter->p_cublas_handle), CUBLAS_OP_N, CUBLAS_OP_N, N, M, 1,
					&a, layer_parameter->meds[0]->gpu_data, N, bias->gpu_data, 1, &a, output->gpu_data+i*out_size, N);

//			data = (float*)malloc(sizeof(float)*output->count);
//			cudaMemcpy(data, output->gpu_data, sizeof(float)*output->count, cudaMemcpyDeviceToHost);
//			fp = fopen("/home/yang/Documents/dl_back_now_change/deeplearning/output.txt", "r");
//			data1 = (float*)malloc(sizeof(float)*output->count);
//			fread(data1, sizeof(float),output->count, fp);
//			for(j = 0; j < output->count; ++j){
//				if(abs(data[j]-data1[j]) < 0.00001){
//					printf("output %d %f %f\n", j, data[j], data1[j]);
//				}
//			}
//			free(data1);
//			free(data);
		}
//		FreeBlob(med);
//		free(med);
	}
//	cudaFree(multi);
}
