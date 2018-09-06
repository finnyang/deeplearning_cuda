#include "DL/util/layer_factory.h"
#include "DL/layers/convolution_layer.h"
#include "DL/layers/input_layer.h"
#include "DL/layers/relu_layer.h"
#include "DL/layers/pooling_layer.h"
#include "DL/layers/innerproduct_layer.h"
#include "DL/layers/softmax_layer.h"
#include "DL/layers/flatten_layer.h"
#include "DL/layers/reshape_layer.h"
#include "DL/layers/permute_layer.h"
#include "DL/layers/concat_layer.h"
#include "DL/layers/priorbox_layer.h"
#include "DL/layers/normalize_layer.h"

void regist_layer(struct Map_Type_Setup_Forward* handle){
	REGISTER_LAYER(convolution);
	REGISTER_LAYER(input);
	REGISTER_LAYER(relu);
	REGISTER_LAYER(pooling);
	REGISTER_LAYER(innerproduct);
	REGISTER_LAYER(softmax);
	REGISTER_LAYER(flatten);
	REGISTER_LAYER(reshape);
	REGISTER_LAYER(permute);
	REGISTER_LAYER(concat);
	REGISTER_LAYER(priorbox);
	REGISTER_LAYER(normalize);
}

/*#include "DL/util/layer_factory.h"
#include "DL/layers/convolution_layer.h"
#include "DL/layers/input_layer.h"
#include "DL/layers/relu_layer.h"
#include "DL/layers/pooling_layer.h"
#include "DL/layers/innerproduct_layer.h"
#include "DL/layers/softmax_layer.h"
#include "DL/layers/flatten_layer.h"
#include "DL/layers/reshape_layer.h"
#include "DL/layers/permute_layer.h"
#include "DL/layers/concat_layer.h"
#include "DL/layers/priorbox_layer.h"
#include "DL/layers/normalize_layer.h"

//需要包含各个层的头文件
void regist_layer(struct Map_Type_Setup_Forward* handle){
	//REGISTER_LAYER(layer_name)
	REGISTER_LAYER(convolution);
	REGISTER_LAYER(input);
	REGISTER_LAYER(relu);
	REGISTER_LAYER(pooling);
	REGISTER_LAYER(innerproduct);
	REGISTER_LAYER(softmax);
	REGISTER_LAYER(flatten);
	REGISTER_LAYER(reshape);
	REGISTER_LAYER(permute);
	REGISTER_LAYER(concat);
	REGISTER_LAYER(priorbox);
	REGISTER_LAYER(normalize);
}*/
