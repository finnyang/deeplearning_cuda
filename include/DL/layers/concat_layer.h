#ifndef CONCAT_LAYER_H_
#define CONCAT_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_concat(struct LayerParameter* layer_parameter);
void forward_concat(struct LayerParameter* layer_parameter);

#endif
