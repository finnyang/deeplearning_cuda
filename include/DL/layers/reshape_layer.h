#ifndef RESHAPE_LAYER_H_
#define RESHAPE_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_reshape(struct LayerParameter* layer_parameter);
void forward_reshape(struct LayerParameter* layer_parameter);

#endif
