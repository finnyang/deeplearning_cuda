#include "DL/util/blob.h"
#include <malloc.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void MakeBlob(int n, int c, int h, int w, struct Blob* blob){
	blob->n = n;
	blob->c = c;
	blob->h = h;
	blob->w = w;
	blob->own = 1;
	blob->count = n*c*h*w;
	cudaMalloc(&(blob->gpu_data), sizeof(float)*blob->count);
}
void FreeBlob(struct Blob* blob){
	if (blob->gpu_data != NULL && blob->own){
		blob->n = 0;
		blob->c = 0;
		blob->h = 0;
		blob->w = 0;
		blob->count = 0;
		cudaFree(blob->gpu_data);
	}
}
