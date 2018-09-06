#include "DL/layers/innerproduct_layer.h"
#include <cblas.h>
#include <malloc.h>
#include <stdlib.h>
#include "DL/util/common_function.h"

void setup_innerproduct(struct LayerParameter* layer_parameter){
	struct InnerproductParameter innerproduct_param = layer_parameter->parameter.innerproduct_param;
	int num_out = innerproduct_param.num_output;
	struct Blob* output = layer_parameter->tops[0];
	struct Blob* input = layer_parameter->bottoms[0];
	int num_in = input->count/input->n;
	MakeBlob(input->n,num_out, 1, 1, output);

	struct Blob* weights = layer_parameter->learn_parameter[0];
	struct Blob* bias = layer_parameter->learn_parameter[1];
	MakeBlob(num_out, num_in, 1, 1, weights);
	MakeBlob(num_out, 1, 1, 1, bias);
	MakeBlob(input->n, 1, 1, 1, layer_parameter->meds[0]);
	set(layer_parameter->meds[0]->gpu_data, input->n, 1.0);
	Doutput_shape_info();
}

void forward_innerproduct(struct LayerParameter* layer_parameter){
	Doutput_info();
	struct Blob* input = layer_parameter->bottoms[0];
	struct Blob* output = layer_parameter->tops[0];
	struct Blob* weights = layer_parameter->learn_parameter[0];
	struct Blob* bias = layer_parameter->learn_parameter[1];
	int num_in = input->count/input->n;
	int num_out = output->c;
	int batch = input->n;
	struct Blob temp;
	MakeBlob(batch, 1, 1, 1, &temp);
	set(temp.gpu_data, batch, 1.0);
	const float a = 1.0;
	const float b = 0.0;
	cublasSgemm(*(layer_parameter->p_cublas_handle), CUBLAS_OP_T, CUBLAS_OP_N, num_out, batch, num_in,
					&a, weights->gpu_data, num_in, input->gpu_data, num_in, &b, output->gpu_data, num_out);
	cublasSgemm(*(layer_parameter->p_cublas_handle), CUBLAS_OP_N, CUBLAS_OP_N, num_out, batch, 1,
					&a, bias->gpu_data, num_out, layer_parameter->meds[0]->gpu_data, 1, &a, output->gpu_data, num_out);
//	cublasSgemm(*(layer_parameter->p_cublas_handle), CUBLAS_OP_N, CUBLAS_OP_N, num_out, batch, 1,
//						&a, bias->gpu_data, num_out, temp.gpu_data, 1, &a, output->gpu_data, num_out);
	FreeBlob(&temp);
}
