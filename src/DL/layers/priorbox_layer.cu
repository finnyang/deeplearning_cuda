#include "DL/layers/priorbox_layer.h"
#include <math.h>
void setup_priorbox(struct LayerParameter* layer_parameter){
	//bottom[0]表示输入的featuremap， bottom[1]表示输入的图像
	int h = layer_parameter->bottoms[0]->h;
	int w = layer_parameter->bottoms[0]->w;
	int shape[4] = {1, 2, h*w*4*layer_parameter->parameter.priorbox_param.priorbox_num, 1};
	MakeBlob(shape[0], shape[1], shape[2], shape[3], layer_parameter->tops[0]);
	Doutput_shape_info();
}

//__global__ void priorbox_kernel(float* top_data, int min_size_, int max_size_, int layer_width, int layer_height,
//		int img_width, int img_height, int step_x, int step_y, int dim, int aspect_num, int num_prior,
//		float* v, float* aspect){
//	int id = blockIdx.x*blockDim.x+threadIdx.x;
//	int h = id/layer_width;
//	int w = id%layer_width;
//	int i,j;
//	if(h < layer_height && w < layer_width){
//		int idx = id*num_prior*4;
//		float center_x,center_y, box_width, box_height;
//		center_x = (w + 0.5) * step_x;
//		center_y = (h + 0.5) * step_y;
//		box_width = box_height = min_size_;
//		top_data[idx++] = (center_x - box_width / 2.) / img_width;
//		top_data[idx++] = (center_y - box_height / 2.) / img_height;
//		top_data[idx++] = (center_x + box_width / 2.) / img_width;
//		top_data[idx++] = (center_y + box_height / 2.) / img_height;
//		if(max_size_ != -1){
//			box_width = box_height = sqrt(float(min_size_ * max_size_));
//			top_data[idx++] = (center_x - box_width / 2.) / img_width;
//			top_data[idx++] = (center_y - box_height / 2.) / img_height;
//			top_data[idx++] = (center_x + box_width / 2.) / img_width;
//			top_data[idx++] = (center_y + box_height / 2.) / img_height;
//		}
//		for(i = 0; i < aspect_num; ++i){
//			box_width = min_size_ * sqrt(aspect[i]);
//			box_height = min_size_ / sqrt(aspect[i]);
//			top_data[idx++] = (center_x - box_width / 2.) / img_width;
//			top_data[idx++] = (center_y - box_height / 2.) / img_height;
//			top_data[idx++] = (center_x + box_width / 2.) / img_width;
//			top_data[idx++] = (center_y + box_height / 2.) / img_height;
//		}
//		idx = id*num_prior*4+dim;
//		for(i = 0; i < num_prior; ++i){
//			for(j = 0; j < 4; ++j){
//				top_data[idx++] = v[j];
//			}
//		}
//	}
//}

void forward_priorbox(struct LayerParameter* layer_parameter){
//	Doutput_info();
//	int min_size_ = layer_parameter->parameter.priorbox_param.min_size;
//	int max_size_ = layer_parameter->parameter.priorbox_param.max_size;
//	int layer_width = layer_parameter->bottoms[0]->w;
//	int layer_height = layer_parameter->bottoms[0]->h;
//	int img_width = layer_parameter->bottoms[1]->w;
//	int img_height = layer_parameter->bottoms[1]->h;
//	float step_x = 1.0*img_width/layer_width;
//	float step_y = 1.0*img_height/layer_height;
//	float* top_data = layer_parameter->tops[0]->gpu_data;
//	int dim = layer_height*layer_width*4*layer_parameter->parameter.priorbox_param.priorbox_num;
////	int idx = 0;
////	int h, w, i, j;
//	int aspect_num = layer_parameter->parameter.priorbox_param.aspect_num;
//	int num_prior = layer_parameter->parameter.priorbox_param.priorbox_num;
//	float* v = layer_parameter->parameter.priorbox_param.variance;
//	float* aspect = layer_parameter->parameter.priorbox_param.aspect;
//	priorbox_kernel<<<(layer_width*layer_height+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(top_data,min_size_, max_size_, layer_width, layer_height,
//			img_width, img_height, step_x, step_y, dim, aspect_num, num_prior,
//			v, aspect);
//	cudaError_t cudaerror = cudaPeekAtLastError();
//	if(cudaerror == cudaSuccess){
//		printf("success\n");
//	}else{
//		printf("%s \n", cudaGetErrorString(cudaerror));
//	}
//	for(;;);

	Doutput_info();
	int min_size_ = layer_parameter->parameter.priorbox_param.min_size;
	int max_size_ = layer_parameter->parameter.priorbox_param.max_size;
	int layer_width = layer_parameter->bottoms[0]->w;
	int layer_height = layer_parameter->bottoms[0]->h;
	int img_width = layer_parameter->bottoms[1]->w;
	int img_height = layer_parameter->bottoms[1]->h;
	float step_x = 1.0*img_width/layer_width;
	float step_y = 1.0*img_height/layer_height;
	float* top_data = (float*)malloc(sizeof(float)*layer_parameter->tops[0]->count);//layer_parameter->tops[0]->cpu_data;
	float* top_data_ori = top_data;
	int dim = layer_height*layer_width*4*layer_parameter->parameter.priorbox_param.priorbox_num;
	int idx = 0;
	int h, w, i, j;
	int aspect_num = layer_parameter->parameter.priorbox_param.aspect_num;
	int num_prior = layer_parameter->parameter.priorbox_param.priorbox_num;
	float* v = layer_parameter->parameter.priorbox_param.variance;
	float center_x,center_y, box_width, box_height;
	for(h = 0; h < layer_height; ++h){
		for(w = 0; w < layer_width; ++w){
			center_x = (w + 0.5) * step_x;
			center_y = (h + 0.5) * step_y;
			box_width = box_height = min_size_;
			top_data[idx++] = (center_x - box_width / 2.) / img_width;
			top_data[idx++] = (center_y - box_height / 2.) / img_height;
			top_data[idx++] = (center_x + box_width / 2.) / img_width;
			top_data[idx++] = (center_y + box_height / 2.) / img_height;
			if(max_size_ != -1){
				box_width = box_height = sqrt(min_size_ * max_size_);
				top_data[idx++] = (center_x - box_width / 2.) / img_width;
				top_data[idx++] = (center_y - box_height / 2.) / img_height;
				top_data[idx++] = (center_x + box_width / 2.) / img_width;
				top_data[idx++] = (center_y + box_height / 2.) / img_height;
			}
			for(i = 0; i < aspect_num; ++i){
				box_width = min_size_ * sqrt(layer_parameter->parameter.priorbox_param.aspect[i]);
				box_height = min_size_ / sqrt(layer_parameter->parameter.priorbox_param.aspect[i]);
				top_data[idx++] = (center_x - box_width / 2.) / img_width;
				top_data[idx++] = (center_y - box_height / 2.) / img_height;
				top_data[idx++] = (center_x + box_width / 2.) / img_width;
				top_data[idx++] = (center_y + box_height / 2.) / img_height;
			}
		}
	}
	for(i = 0; i < dim; ++i){
		if(top_data[i] > 1.0)
			top_data[i] = 1.0;
		else
			if(top_data[i] < 0.0)
				top_data[i] = 0.0;
	}
	top_data += dim;
	idx = 0;
	for(h = 0; h < layer_height; ++h){
		for(w = 0; w < layer_width; ++w){
			for(i = 0; i < num_prior; ++i){
				for(j = 0; j < 4; ++j){
					top_data[idx++] = v[j];
				}
			}
		}
	}
	cudaMemcpy(layer_parameter->tops[0]->gpu_data, top_data_ori, sizeof(float)*layer_parameter->tops[0]->count, cudaMemcpyHostToDevice);
	free(top_data_ori);
}
