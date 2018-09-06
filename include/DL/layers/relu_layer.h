#ifndef RELU_LAYER_H_
#define RELU_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_relu(struct LayerParameter* layer_parameter);
void forward_relu(struct LayerParameter* layer_parameter);

#endif
