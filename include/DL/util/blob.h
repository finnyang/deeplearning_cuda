#ifndef DL_BLOB_H_
#define DL_BLOB_H_

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

struct Blob {
	int n;
	int c;
	int h;
	int w;
	int count;
	int own;
	float* gpu_data;
};

void MakeBlob(int n, int c, int h, int w, struct Blob* blob);
void FreeBlob(struct Blob* blob);

#endif
