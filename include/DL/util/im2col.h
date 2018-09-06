#ifndef IM2COL_H_
#define IM2COL_H_
//void im2col_cpu(float* data_im,
//		int channels, int height, int width,
//		int ksize, int stride, int pad, int dilation,float* data_col);
void im2col_gpu(float* data_im,
		int channels, int height, int width,
		int ksize, int stride, int pad, int dilation, float* data_col);
#endif
