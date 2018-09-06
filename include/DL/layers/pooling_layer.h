#ifndef DL_POOLING_LAYER_H_
#define DL_POOLING_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_pooling(struct LayerParameter* layer_parameter);
void forward_pooling(struct LayerParameter* layer_parameter);

#endif
