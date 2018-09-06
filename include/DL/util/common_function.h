#ifndef COMMON_FUNCTION_H_
#define COMMON_FUNCTION_H_

#include "DL/util/blob.h"
#include "DL/util/layer.h"
#include <stdlib.h>

void set(float* input, int count, float scale);
void copy(float* input, float* output, int count);
void exp_gpu(float* input, float* output, int count);
void power_gpu(float* input, float* output, int count);
void scale_gpu(float* input, float* output, int count, float scale);

#endif
