#ifndef PRIORBOX_LAYER_H_
#define PRIORBOX_LAYER_H_

#include <stdio.h>
#include "DL/util/layer.h"
#include "DL/util/blob.h"

void setup_priorbox(struct LayerParameter* layer_parameter);
void forward_priorbox(struct LayerParameter* layer_parameter);

#endif
