#ifndef INNERPRODUCT_LAYER_
#define INNERPRODUCT_LAYER_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_innerproduct(struct LayerParameter* layer_parameter);
void forward_innerproduct(struct LayerParameter* layer_parameter);

#endif
