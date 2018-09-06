#include "DL/layers/flatten_layer.h"

void setup_flatten(struct LayerParameter* layer_parameter){
	int start_axis = layer_parameter->parameter.flatten_param.start_axis;
	int end_axis = layer_parameter->parameter.flatten_param.end_axis;
	int i;
	int idx=1;
	int temp = 1;
	int shape_ori[4] = {layer_parameter->bottoms[0]->n, layer_parameter->bottoms[0]->c, layer_parameter->bottoms[0]->h, layer_parameter->bottoms[0]->w};
	int shape[4]={1, 1, 1, 1};
	for(i = 0; i < start_axis; ++i){
		shape[i] = shape_ori[i];
	}
	for(i = start_axis; i < end_axis+1; ++i){
		temp*=shape_ori[i];
	}
	shape[start_axis] = temp;
	for(i = end_axis+1; i < 4; ++i){
		shape[start_axis+idx] = shape_ori[i];
		++idx;
	}
	layer_parameter->tops[0]->n = shape[0];
	layer_parameter->tops[0]->c = shape[1];
	layer_parameter->tops[0]->h = shape[2];
	layer_parameter->tops[0]->w = shape[3];
	layer_parameter->tops[0]->gpu_data = layer_parameter->bottoms[0]->gpu_data;
	layer_parameter->tops[0]->count = layer_parameter->bottoms[0]->count;
	layer_parameter->tops[0]->own = 0;
	Doutput_shape_info();
}
void forward_flatten(struct LayerParameter* layer_parameter){
	Doutput_info();
}
