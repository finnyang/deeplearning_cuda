#ifndef DL_NET_H_
#define DL_NET_H_

#include "DL/util/layer.h"
#include "DL/util/blob.h"
#include "DL/util/layer_factory.h"

struct Net {
	cublasHandle_t cublas_handle;
	int layer_num;
	int tops_num;
	struct Blob** tops;
	int input_num;
	int output_num;
	struct Blob** inputs;
	struct Blob** outputs;
	char** names;
	struct LayerParameter* layer_parameter;
	struct Map_Type_Setup_Forward handle;
};

void forward(struct Net net);//向前传播
void setup(struct Net net);//网络的层创建
void deletenet(struct Net net);

void make_convolution(struct Net* net, char* name,int bottom_size, char** bottom_name, int top_size, char** top_name,
		int num_output, int pad, int stride, int kernel_size, int dilation);
void make_input(struct Net* net, char* name, int top_size, char** top_name,
		int n, int c, int h, int w);
void make_relu(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		float negative_slope);
void make_pooling(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int pad, int stride, int kernel_size, enum PoolingType pooling_type, int global_pooling);
void make_innerproduct(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int num_output);
void make_softmax(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int axis);
void make_flatten(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int start_axis, int end_axis);
void make_reshape(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int dim_n, int dim_c, int dim_h, int dim_w);
void make_permute(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int idx_n, int idx_c, int idx_h, int idx_w);
void make_concat(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int axis);
void make_priorbox(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int min_size, int max_size, float aspect1, float aspect2, float v1, float v2, float v3, float v4);
void make_normalize(struct Net* net, char* name, int bottom_size, char** bottom_name, int top_size, char** top_name,
		int channel_shared, int across_spatial);
void set_net_io(struct Net* net, int input_size, char** input_name, int output_size, char** output_name);

//void InitLayerName(struct LayerName* layername);
//
//void DeleteLayerName(struct LayerName* layername);

#define init() \
int i,j;\
(net->layer_num) += 1;\
(net->tops_num) += top_size;\
net->layer_parameter = (struct LayerParameter*)realloc(net->layer_parameter, sizeof(struct LayerParameter)*net->layer_num);\
struct LayerParameter* new_layer = net->layer_parameter+(net->layer_num-1);\
new_layer->name = name;\
new_layer->bottom_num = bottom_size;\
new_layer->top_num = top_size;\
new_layer->bottom = (char**)malloc(sizeof(char*)*bottom_size);\
new_layer->top = (char**)malloc(sizeof(char*)*top_size);\
new_layer->bottoms = (struct Blob**)malloc(sizeof(struct Blob*)*bottom_size);\
new_layer->tops = (struct Blob**)malloc(sizeof(struct Blob*)*top_size);\
net->tops = (struct Blob**)realloc(net->tops, sizeof(struct Blob*)*net->tops_num);\
net->names = (char**)realloc(net->names, sizeof(char*)*net->tops_num);\
new_layer->p_cublas_handle = &(net->cublas_handle);\
for(i = 0; i < top_size; ++i){\
	net->names[net->tops_num-top_size+i] = top_name[i];\
	new_layer->top[i] = top_name[i];\
	net->tops[net->tops_num-top_size+i] = (struct Blob*)malloc(sizeof(struct Blob));\
	new_layer->tops[i]=(net->tops[net->tops_num-top_size+i]);\
}\
for(i = 0; i < bottom_size; ++i){\
	new_layer->bottom[i] = bottom_name[i];\
	for(j = 0; j < net->tops_num-top_size; ++j){\
		if(strcmp(bottom_name[i], net->names[j]) == 0){\
			new_layer->bottoms[i] = (net->tops[j]);\
		}\
	}\
}
#endif
