#include "DL/layers/input_layer.h"

void setup_input(struct LayerParameter* layer_parameter) {
	struct InputParameter input_param = layer_parameter->parameter.input_param;
	MakeBlob(input_param.n, input_param.c, input_param.h, input_param.w, layer_parameter->tops[0]);
	Doutput_shape_info();
}
void forward_input(struct LayerParameter* layer_parameter) {
	Doutput_info();
}
