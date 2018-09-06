#ifndef CONVOLUTION_LAYER_H_
#define CONVOLUTION_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_convolution(struct LayerParameter* layer_parameter);
void forward_convolution(struct LayerParameter* layer_parameter);

#endif
