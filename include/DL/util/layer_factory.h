#ifndef DL_LAYER_FACTORY_H_
#define DL_LAYER_FACTORY_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include <stdlib.h>

typedef void (*function)(struct LayerParameter*);

struct Type_Setup_Forward {
	char* type;
	function setup;
	function forward_cpu;
};

struct Map_Type_Setup_Forward {
	int num;
	struct Type_Setup_Forward* functions;
};

void regist_layer(struct Map_Type_Setup_Forward* handle);

//自动注册的代码,每写完一个层都要在注册函里面添加注册的宏定义。
#define REGISTER_LAYER(name)\
handle->num = handle->num+1;\
handle->functions = (struct Type_Setup_Forward*)realloc(handle->functions, sizeof(struct Type_Setup_Forward)*handle->num);\
handle->functions[handle->num-1].type=#name;\
handle->functions[handle->num-1].setup=setup_##name;\
handle->functions[handle->num-1].forward_cpu=forward_##name;

#endif
