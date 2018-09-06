#include "DL/layers/reshape_layer.h"

void setup_reshape(struct LayerParameter* layer_parameter){
	int dim_n = layer_parameter->parameter.reshape_param.dim_n;
	int dim_c = layer_parameter->parameter.reshape_param.dim_c;
	int dim_h = layer_parameter->parameter.reshape_param.dim_h;
	int dim_w = layer_parameter->parameter.reshape_param.dim_w;
	int n = layer_parameter->bottoms[0]->n;
	int c = layer_parameter->bottoms[0]->c;
	int h = layer_parameter->bottoms[0]->h;
	int w = layer_parameter->bottoms[0]->w;
	int count = n*c*h*w;

	if(dim_n == 0) dim_n = n;
	if(dim_c == 0) dim_c = c;
	if(dim_h == 0) dim_h = h;
	if(dim_w == 0) dim_w = w;

	if(dim_n == -1) dim_n = count/(dim_w*dim_h*dim_c);
	if(dim_c == -1) dim_c = count/(dim_n*dim_h*dim_w);
	if(dim_h == -1) dim_h = count/(dim_n*dim_c*dim_w);
	if(dim_w == -1) dim_w = count/(dim_n*dim_c*dim_h);

	layer_parameter->tops[0]->n = dim_n;
	layer_parameter->tops[0]->c = dim_c;
	layer_parameter->tops[0]->h = dim_h;
	layer_parameter->tops[0]->w = dim_w;
	layer_parameter->tops[0]->gpu_data = layer_parameter->bottoms[0]->gpu_data;
	layer_parameter->tops[0]->count = layer_parameter->bottoms[0]->count;
	layer_parameter->tops[0]->own = 0;
	Doutput_shape_info();
}
void forward_reshape(struct LayerParameter* layer_parameter){
	Doutput_info();
}
