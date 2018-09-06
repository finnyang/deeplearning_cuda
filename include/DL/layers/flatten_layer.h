#ifndef FLATTEN_LAYER_H_
#define FLATTEN_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_flatten(struct LayerParameter* layer_parameter);
void forward_flatten(struct LayerParameter* layer_parameter);

#endif
