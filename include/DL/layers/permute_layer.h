#ifndef PERMUTE_LAYER_H_
#define PERMUTE_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_permute(struct LayerParameter* layer_parameter);
void forward_permute(struct LayerParameter* layer_parameter);

#endif
