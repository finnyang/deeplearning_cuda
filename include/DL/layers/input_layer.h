#ifndef DL_INPUT_LAYER_
#define DL_INPUT_LAYER_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_input(struct LayerParameter* layer_parameter);
void forward_input(struct LayerParameter* layer_parameter);

#endif
