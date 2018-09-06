#ifndef SOFTMAX_LAYER_H_
#define SOFTMAX_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_softmax(struct LayerParameter* layer_parameter);
void forward_softmax(struct LayerParameter* layer_parameter);

#endif
