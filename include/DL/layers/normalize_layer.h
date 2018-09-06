#ifndef NORMALIZE_LAYER_H_
#define NORMALIZE_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_normalize(struct LayerParameter* layer_parameter);
void forward_normalize(struct LayerParameter* layer_parameter);

#endif
