#ifndef DL_LAYER_H_
#define DL_LAYER_H_

#include <stdio.h>
#include "DL/util/blob.h"
#include <stdlib.h>

const int CUDA_NUM_THREADS=512;

//层的注册顺序要跟LayerType里面的顺序一致，这样做的好处是为了减少不必要的复杂度， LayerName的顺序也是同样的。
//注册的位置在layer_factory.c regist_layer函数，注册的注意事项写明在layer_factory.h中
//LayerName的位置在Net.c InitLayerName函数
enum LayerType {
	CONVOLUTION,
	INPUT,
	RELU,
	POOLING,
	INNERPRODUCT,
	SOFTMAX,
	FLATTEN,
	RESHAPE,
	PERMUTE,
	CONCAT,
	PRIORBOX,
	NORMALIZE
};;
/*
enum LayerType {
	CONVOLUTION,
	INPUT,
	RELU,
	POOLING,
	INNERPRODUCT,
	SOFTMAX,
	FLATTEN,
	RESHAPE,
	PERMUTE,
	CONCAT,
	PRIORBOX,
	NORMALIZE
};*/
enum PoolingType{
	MAX,
	AVE
};

//struct LayerName {
//	char** names;
//};

struct ConvolutionParameter {
	int num_output;
	int pad;
	int stride;
	int kernel_size;
	int dilation;
};

struct InputParameter {
	int n;
	int c;
	int h;
	int w;
};

struct ReluParameter {
	float negative_slope;
};

struct InnerproductParameter {
	int num_output;
};

struct PoolingParameter {
	int pad;
	int stride;
	int kernel_size;
	enum PoolingType pooling_type;
	int global_pooling;
};

struct SoftmaxParameter {
	int axis;
};

struct FlattenParameter {
	int start_axis;
	int end_axis;
};

struct ReshapeParameter{
	int dim_n;
	int dim_c;
	int dim_h;
	int dim_w;
};

struct PermuteParameter{
	int idx_n;
	int idx_c;
	int idx_h;
	int idx_w;
};

struct ConcatParameter{
	int axis;
};

struct PriorboxParameter {
	int min_size;//必须要进行设置
	int max_size;//max_size 如果没有，需要设置为-1
	float aspect[6];
	int aspect_num;
	int priorbox_num;
	float variance[4];
};

struct NormalizeParameter {
	int channel_shared;
	int across_spatial;
};

union Parameter {
	struct ConvolutionParameter convolution_param;
	struct InputParameter input_param;
	struct ReluParameter relu_param;
	struct PoolingParameter pooling_param;
	struct InnerproductParameter innerproduct_param;
	struct SoftmaxParameter softmax_param;
	struct FlattenParameter flatten_param;
	struct ReshapeParameter reshape_param;
	struct PermuteParameter permute_param;
	struct ConcatParameter concat_param;
	struct PriorboxParameter priorbox_param;
	struct NormalizeParameter normalize_param;
};

struct LayerParameter {
	char* name;//层的名称
	int bottom_num;
	int top_num;
	char** bottom;//层bottom的名称
	char** top;//层top的名称
	enum LayerType type;//层的类型
	union Parameter parameter;//层的参数（不可学习参数）
	int has_learn_parameter;//表示是否学习的参数，如果有，其中的数表示参数类型的个数，比如（w，b，则它被赋值为2）
	//使用指针的指针是为了应对多个bottom、多个top或多个学习参数，这里只需要指向指针的指针是因为在net里面统一管理这些空间。指针的指针需要释放空间。
	struct Blob** learn_parameter;//层的可学习参数
	struct Blob** bottoms;//层的输入
	struct Blob** tops;//层的输出
	cublasHandle_t* p_cublas_handle;
	int med_num;
	struct Blob** meds;//用于计算的中间变量
};

#ifdef DEBUG
#define Doutput_info() \
printf("forward %s\n", layer_parameter->name)
#else
#define Doutput_info() ;
#endif

#ifdef DEBUG
#define Doutput_shape_info() \
printf("setup %s\n", layer_parameter->name);\
int TEMP_; \
for(TEMP_ = 0; TEMP_ < layer_parameter->bottom_num; ++TEMP_){ \
	printf("input (%d,%d,%d,%d)\n", layer_parameter->bottoms[TEMP_]->n, layer_parameter->bottoms[TEMP_]->c, layer_parameter->bottoms[TEMP_]->h, layer_parameter->bottoms[TEMP_]->w); \
}\
for(TEMP_ = 0; TEMP_ < layer_parameter->top_num; ++TEMP_){ \
	printf("output (%d,%d,%d,%d)\n\n",layer_parameter->tops[TEMP_]->n, layer_parameter->tops[TEMP_]->c, layer_parameter->tops[TEMP_]->h, layer_parameter->tops[TEMP_]->w); \
}
#else
#define Doutput_shape_info() ;
#endif

#endif
