#include "DL/util/net.h"
#include <stdlib.h>
#include <string.h>

 void make_convolution(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int num_output, int pad, int stride, int kernel_size, int dilation){
	init();
	new_layer->type = CONVOLUTION;
	struct ConvolutionParameter* convolution_param = &(new_layer->parameter.convolution_param);
	convolution_param->kernel_size = kernel_size;
	convolution_param->num_output = num_output;
	convolution_param->pad = pad;
	convolution_param->stride = stride;
	convolution_param->dilation = dilation;
	new_layer->has_learn_parameter = 2;
	new_layer->learn_parameter = (struct Blob**)malloc(sizeof(struct Blob*)*2);
	new_layer->learn_parameter[0] = (struct Blob*)malloc(sizeof(struct Blob));
	new_layer->learn_parameter[1] = (struct Blob*)malloc(sizeof(struct Blob));
	if(pad == 0 && dilation == 1 && stride == 1){
		new_layer->med_num = 1;
		new_layer->meds = (struct Blob**)malloc(sizeof(struct Blob*));
		new_layer->meds[0] = (struct Blob*)malloc(sizeof(struct Blob));
	}else{
		new_layer->med_num = 2;
		new_layer->meds = (struct Blob**)malloc(sizeof(struct Blob*)*2);
		new_layer->meds[0] = (struct Blob*)malloc(sizeof(struct Blob));
		new_layer->meds[1] = (struct Blob*)malloc(sizeof(struct Blob));
	}
}

void make_input(struct Net* net, char* name, int top_size, char** top_name,
		int n, int c, int h, int w){
    int i;
	net->layer_num += 1;
	net->tops_num += top_size;
	net->layer_parameter = (struct LayerParameter*)realloc(net->layer_parameter, sizeof(struct LayerParameter)*net->layer_num);
	struct LayerParameter* new_layer = net->layer_parameter+(net->layer_num-1);
	new_layer->name = name;
    new_layer->bottom_num = 0;
	new_layer->top_num = top_size;
	new_layer->bottom = NULL;
	new_layer->top = (char**)malloc(sizeof(char*)*top_size);
	new_layer->bottoms = NULL;
	new_layer->tops = (struct Blob**)malloc(sizeof(struct Blob*)*top_size);
	net->tops = (struct Blob**)realloc(net->tops, sizeof(struct Blob*)*net->tops_num);
	net->names = (char**)realloc(net->names, sizeof(char*)*net->tops_num);
	for(i = 0; i < top_size; ++i){
	  net->names[net->tops_num-top_size+i] = top_name[i];
	  new_layer->top[i] = top_name[i];
	  net->tops[net->tops_num-top_size+i] = (struct Blob*)malloc(sizeof(struct Blob));
	  new_layer->tops[i]=(net->tops[net->tops_num-top_size+i]);
	}
	new_layer->type = INPUT;
	struct InputParameter* input_param = &(new_layer->parameter.input_param);
	input_param->n = n;
	input_param->c = c;
	input_param->h = h;
	input_param->w = w;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
	new_layer->med_num = 0;
	new_layer->meds = NULL;
}

void make_relu(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		float negative_slope){
	init();
	new_layer->type = RELU;
	struct ReluParameter* relu_param = &(new_layer->parameter.relu_param);
	relu_param->negative_slope = negative_slope;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
	new_layer->med_num = 0;
	new_layer->meds = NULL;
}

void make_pooling(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int pad, int stride, int kernel_size, enum PoolingType pooling_type, int global_pooling){
    init();
	new_layer->type = POOLING;
	struct PoolingParameter* pooling_param = &(new_layer->parameter.pooling_param);
	pooling_param->kernel_size = kernel_size;
	pooling_param->pad = pad;
	pooling_param->stride = stride;
	pooling_param->pooling_type = pooling_type;
	pooling_param->global_pooling = global_pooling;
	if(global_pooling){
		pooling_param->pad = 0;
		pooling_param->stride = 1;
	}
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
	new_layer->med_num = 0;
	new_layer->meds = NULL;
}

void make_innerproduct(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int num_output){
    init();
	new_layer->type = INNERPRODUCT;
	struct InnerproductParameter* innerproduct_param = &(new_layer->parameter.innerproduct_param);
	innerproduct_param->num_output = num_output;
	new_layer->has_learn_parameter = 2;
	new_layer->learn_parameter = (struct Blob**)malloc(sizeof(struct Blob*)*2);
	new_layer->learn_parameter[0] = (struct Blob*)malloc(sizeof(struct Blob));
	new_layer->learn_parameter[1] = (struct Blob*)malloc(sizeof(struct Blob));
	new_layer->med_num = 1;
	new_layer->meds =(struct Blob**)malloc(sizeof(struct Blob*));
	new_layer->meds[0] = (struct Blob*)malloc(sizeof(struct Blob));
//	new_layer->med_num = 0;
//	new_layer->meds = NULL;
}

void make_softmax(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char**top_name,
		int axis){
	init();
	new_layer->type = SOFTMAX;
	struct SoftmaxParameter* softmax_param = &(new_layer->parameter.softmax_param);//根据层特有的参数来确定
	softmax_param->axis = axis;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
	new_layer->med_num = 1;
	new_layer->meds =(struct Blob**)malloc(sizeof(struct Blob*));
	new_layer->meds[0] = (struct Blob*)malloc(sizeof(struct Blob));
}

void make_permute(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int idx_n, int idx_c, int idx_h, int idx_w){
	init();
	new_layer->type = PERMUTE;
	struct PermuteParameter* permute_param = &(new_layer->parameter.permute_param);
	permute_param->idx_n = idx_n;
	permute_param->idx_c = idx_c;
	permute_param->idx_h = idx_h;
	permute_param->idx_w = idx_w;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
	new_layer->med_num = 0;
	new_layer->meds = NULL;
}

void make_flatten(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int start_axis, int end_axis){
	init();
	new_layer->type = FLATTEN;
	struct FlattenParameter* flatten_param = &(new_layer->parameter.flatten_param);;
	flatten_param->start_axis = start_axis;
	flatten_param->end_axis = end_axis;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
	new_layer->med_num = 0;
	new_layer->meds = NULL;
}

void make_reshape(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int dim_n, int dim_c, int dim_h, int dim_w){
    init();
	new_layer->type = RESHAPE;
	struct ReshapeParameter* reshape_param = &(new_layer->parameter.reshape_param);
	reshape_param->dim_n = dim_n;
	reshape_param->dim_c = dim_c;
	reshape_param->dim_h = dim_h;
	reshape_param->dim_w = dim_w;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
	new_layer->med_num = 0;
	new_layer->meds = NULL;
}

void make_concat(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int axis){
	init();
	new_layer->type = CONCAT;
	struct ConcatParameter* concat_param = &(new_layer->parameter.concat_param);
	concat_param->axis = axis;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
	new_layer->med_num = 0;
	new_layer->meds = NULL;
}

void make_priorbox(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int min_size, int max_size, float aspect1, float aspect2, float v1, float v2, float v3, float v4){
	init();
	new_layer->type = PRIORBOX;
	struct PriorboxParameter* priorbox_param = &(new_layer->parameter.priorbox_param);
	priorbox_param->min_size = min_size;
	priorbox_param->max_size = max_size;
	priorbox_param->variance[0] = v1;
	priorbox_param->variance[1] = v2;
	priorbox_param->variance[2] = v3;
	priorbox_param->variance[3] = v4;
	i = 3;
	j = 2;
	priorbox_param->aspect[0] = aspect1;
	priorbox_param->aspect[1] = 1/aspect1;
	if(aspect2 > 0){
		priorbox_param->aspect[2] = aspect2;
		priorbox_param->aspect[3] = 1/aspect2;
		i+=2;
		j+=2;
	}
	if(max_size > 0){
		++i;
	}
	priorbox_param->priorbox_num = i;
	priorbox_param->aspect_num = j;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
	new_layer->med_num = 0;
	new_layer->meds = NULL;
}

void make_normalize(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int channel_shared, int across_spatial) {
	init();
	new_layer->type = NORMALIZE;
	struct NormalizeParameter* normalize_param = &(new_layer->parameter.normalize_param);
	normalize_param->across_spatial = across_spatial;
	normalize_param->channel_shared = channel_shared;
	new_layer->has_learn_parameter = 1;
	new_layer->learn_parameter = (struct Blob**)malloc(sizeof(struct Blob*));
	new_layer->learn_parameter[0] = (struct Blob*)malloc(sizeof(struct Blob));
	new_layer->med_num = 0;
	new_layer->meds = NULL;
}

void set_net_io(struct Net* net, int input_size, char** input_name, int output_size, char** output_name){
	int i,j;
	net->input_num = input_size;
	net->output_num = output_size;
	net->inputs = (struct Blob**)malloc(sizeof(struct Blob*)*input_size);
	net->outputs = (struct Blob**)malloc(sizeof(struct Blob*)*output_size);
	for(i = 0; i < input_size; ++i){
		for(j = 0; j < net->tops_num; ++j){
			if(strcmp(input_name[i], net->names[j]) == 0){
				net->inputs[i] = net->tops[j];
			}
		}
	}
	for(i = 0; i < output_size; ++i){
		for(j = 0; j < net->tops_num; ++j){
			if(strcmp(output_name[i], net->names[j]) == 0){
				net->outputs[i] = net->tops[j];
			}
		}
	}
}

//void InitLayerName(struct LayerName* layername){
//	//num表示当前层的个数
//	int num = 3;
//	int i = 0;
//    layername->names = NULL;
//	layername->names = (char**)realloc(layername->names, sizeof(char*)*num);
//	layername->names[i++] = "input";
//	layername->names[i++] = "relu";
//    layername->names[i++] = "flatten";
//}

//void DeleteLayerName(struct LayerName* layername){
//	free(layername->names);
//}

void forward(struct Net net){
  int i = 0;
  for(i = 0; i < net.layer_num; ++i){
    net.handle.functions[net.layer_parameter[i].type].forward_cpu(net.layer_parameter+i);
  }
}

void setup(struct Net net) {
  int i = 0;
  for(i = 0; i < net.layer_num; ++i){
    net.handle.functions[net.layer_parameter[i].type].setup(net.layer_parameter+i);
  }
}

void deletenet(struct Net net){//网络的内存释放
	int i, j;
	cublasDestroy(net.cublas_handle);
	free(net.inputs);
	free(net.outputs);
	for (i = 0; i < net.tops_num; ++i){
		FreeBlob(net.tops[i]);
		free(net.tops[i]);
	}
	free(net.tops);
	free(net.names);
	for(i = 0; i < net.layer_num; ++i){
		free(net.layer_parameter[i].bottom);
		free(net.layer_parameter[i].top);
		free(net.layer_parameter[i].bottoms);
		free(net.layer_parameter[i].tops);
		if(net.layer_parameter[i].has_learn_parameter){
			for(j = 0; j < net.layer_parameter[i].has_learn_parameter; ++j){
				FreeBlob(net.layer_parameter[i].learn_parameter[j]);
				free(net.layer_parameter[i].learn_parameter[j]);
			}
			free(net.layer_parameter[i].learn_parameter);
		}
		if(net.layer_parameter[i].med_num){
#ifdef DEBUG
			printf("realease %s\n", net.layer_parameter[i].name);
#endif
			for(j = 0; j < net.layer_parameter[i].med_num; ++j){
				FreeBlob(net.layer_parameter[i].meds[j]);
				free(net.layer_parameter[i].meds[j]);
			}
			free(net.layer_parameter[i].meds);
		}
	}
	free(net.layer_parameter);
	free(net.handle.functions);
}
/*

void make_innerproduct(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int num_output){
    init();
	new_layer->type = INNERPRODUCT;
	struct InnerproductParameter* innerproduct_param = &(new_layer->parameter.innerproduct_param);
	innerproduct_param->num_output = num_output;
	new_layer->has_learn_parameter = 2;
	new_layer->learn_parameter = (struct Blob**)malloc(sizeof(struct Blob*)*2);
	new_layer->learn_parameter[0] = (struct Blob*)malloc(sizeof(struct Blob));
	new_layer->learn_parameter[1] = (struct Blob*)malloc(sizeof(struct Blob));
}

void make_softmax(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char**top_name,
		int axis){
	init();
	new_layer->type = SOFTMAX;
	struct SoftmaxParameter* softmax_param = &(new_layer->parameter.softmax_param);//根据层特有的参数来确定
	softmax_param->axis = axis;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
}



void make_permute(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int idx_n, int idx_c, int idx_h, int idx_w){
	init();
	new_layer->type = PERMUTE;
	struct PermuteParameter* permute_param = &(new_layer->parameter.permute_param);
	permute_param->idx_n = idx_n;
	permute_param->idx_c = idx_c;
	permute_param->idx_h = idx_h;
	permute_param->idx_w = idx_w;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
}

void make_concat(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int axis){
	init();
	new_layer->type = CONCAT;
	struct ConcatParameter* concat_param = &(new_layer->parameter.concat_param);
	concat_param->axis = axis;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
}

void make_priorbox(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int min_size, int max_size, float aspect1, float aspect2, float v1, float v2, float v3, float v4){
	init();
	new_layer->type = PRIORBOX;
	struct PriorboxParameter* priorbox_param = &(new_layer->parameter.priorbox_param);
	priorbox_param->min_size = min_size;
	priorbox_param->max_size = max_size;
	priorbox_param->variance[0] = v1;
	priorbox_param->variance[1] = v2;
	priorbox_param->variance[2] = v3;
	priorbox_param->variance[3] = v4;
	i = 3;
	j = 2;
	priorbox_param->aspect[0] = aspect1;
	priorbox_param->aspect[1] = 1/aspect1;
	if(aspect2 > 0){
		priorbox_param->aspect[2] = aspect2;
		priorbox_param->aspect[3] = 1/aspect2;
		i+=2;
		j+=2;
	}
	if(max_size > 0){
		++i;
	}
	priorbox_param->priorbox_num = i;
	priorbox_param->aspect_num = j;
	new_layer->has_learn_parameter = 0;
	new_layer->learn_parameter = NULL;
}

void make_normalize(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int channel_shared, int across_spatial) {
	init();
	new_layer->type = NORMALIZE;
	struct NormalizeParameter* normalize_param = &(new_layer->parameter.normalize_param);
	normalize_param->across_spatial = across_spatial;
	normalize_param->channel_shared = channel_shared;
	new_layer->has_learn_parameter = 1;
	new_layer->learn_parameter = (struct Blob**)malloc(sizeof(struct Blob*)*1);
	new_layer->learn_parameter[0] = (struct Blob*)malloc(sizeof(struct Blob));
}
 */
