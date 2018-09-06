#include "DL/util/im2col.h"
#include "DL/util/layer.h"
//void im2col_cpu(float* data_im,
//		int channels, int height, int width,
//		int ksize, int stride, int pad, int dilation, float* data_col){
//    int c,h,w;
//    int height_col = (height - (dilation*(ksize-1)+1) + 2*pad) / stride + 1;
//    int width_col = (width - (dilation*(ksize-1)+1) + 2*pad) / stride + 1;
//    int channels_col = channels * ksize * ksize;
//    for(c = 0; c < channels_col; ++c){
//        int w_offset = c % ksize;
//        int h_offset = (c / ksize) % ksize;
//        int c_im = c / ksize / ksize;
//        for(h = 0; h < height_col; ++h){
//        	for(w = 0; w < width_col; ++w){
//        		int im_row = h_offset*dilation+h*stride-pad;
//        		int im_col = w_offset*dilation+w*stride-pad;
//        		int col_index = (c * height_col + h) * width_col + w;
//				if( im_row < 0 || im_col < 0 || im_row > height-1 || im_col > width-1)
//					data_col[col_index] = 0.0;
//				else{
//					data_col[col_index] = data_im[(c_im*height+im_row)*width+im_col];
//				}
//        	}
//        }
//    }
//}
/*__global__ void im2col_gpu_kernel(float* data_im,
		int channels, int height, int width,
		int ksize, int stride, int pad, int dilation, float* data_col, int height_col, int width_col, int channels_col){
	int idx_idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx_idx < channels_col){
		int w_offset = idx_idx % ksize;
		int h_offset = (idx_idx / ksize) % ksize;
		int c_im = idx_idx / ksize / ksize;
		int w, h;
		for(h = 0; h < height_col; ++h){
			for(w = 0; w < width_col; ++w){
				int im_row = h_offset*dilation+h*stride-pad;
				int im_col = w_offset*dilation+w*stride-pad;
				int col_index = (idx_idx * height_col + h) * width_col + w;
				if( im_row < 0 || im_col < 0 || im_row > height-1 || im_col > width-1)
					data_col[col_index] = 0.0;
				else{
					data_col[col_index] = data_im[(c_im*height+im_row)*width+im_col];
				}
			}
		}
	}else{
		return;
	}

}*/
__global__ void im2col_gpu_kernel(float* data_im,
		int channels, int height, int width,
		int ksize, int stride, int pad, int dilation, float* data_col, int height_col, int width_col, int channels_col){
//	int idx_idx = blockDim.x*blockIdx.x+threadIdx.x;
//	if(idx_idx < channels_col){
//		int w_offset = idx_idx % ksize;
//		int h_offset = (idx_idx / ksize) % ksize;
//		int c_im = idx_idx / ksize / ksize;
//		int w, h;
//		for(h = 0; h < height_col; ++h){
//			for(w = 0; w < width_col; ++w){
//				int im_row = h_offset*dilation+h*stride-pad;
//				int im_col = w_offset*dilation+w*stride-pad;
//				int col_index = (idx_idx * height_col + h) * width_col + w;
//				if( im_row < 0 || im_col < 0 || im_row > height-1 || im_col > width-1)
//					data_col[col_index] = 0.0;
//				else{
//					data_col[col_index] = data_im[(c_im*height+im_row)*width+im_col];
//				}
//			}
//		}
//	}else{
//		return;
//	}

	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx < height_col*width_col*channels_col){
		int idx_idx = idx/(height_col*width_col);
		int temp = idx%(height_col*width_col);
		int w_offset = idx_idx % ksize;
		int h_offset = (idx_idx / ksize) % ksize;
		int c_im = idx_idx / ksize / ksize;
		int h = temp/width_col;
		int w = temp%width_col;
		int im_row = h_offset*dilation+h*stride-pad;
		int im_col = w_offset*dilation+w*stride-pad;
		int col_index = (idx_idx * height_col + h) * width_col + w;
		if( im_row < 0 || im_col < 0 || im_row > height-1 || im_col > width-1)
			data_col[col_index] = 0.0;
		else{
			data_col[col_index] = data_im[(c_im*height+im_row)*width+im_col];
		}
	}
}

__global__ void im2col_gpu_kernel_new(float* data_im,
		int channels, int height, int width,
		int ksize, int stride, int pad, int dilation, float* data_col, int height_col, int width_col, int channels_col){
	int w_offset = blockIdx.y;
	int h_offset = blockIdx.x;
	int c_im = blockIdx.z;
	int h = threadIdx.x;
	int w = threadIdx.y;

	if(w_offset < ksize && h_offset < ksize && c_im < channels && h < height_col && w < width_col){
		int idx_idx = ((c_im*ksize)+h_offset)*ksize+w_offset;
		int im_row = h_offset*dilation+h*stride-pad;
		int im_col = w_offset*dilation+w*stride-pad;
		int col_index = (idx_idx * height_col + h) * width_col + w;
		if( im_row < 0 || im_col < 0 || im_row > height-1 || im_col > width-1)
			data_col[col_index] = 0.0;
		else{
			data_col[col_index] = data_im[(c_im*height+im_row)*width+im_col];
		}
	}
}

void im2col_gpu(float* data_im,
		int channels, int height, int width,
		int ksize, int stride, int pad, int dilation, float* data_col){
    int height_col = (height - (dilation*(ksize-1)+1) + 2*pad) / stride + 1;
    int width_col = (width - (dilation*(ksize-1)+1) + 2*pad) / stride + 1;
    int channels_col = channels * ksize * ksize;
    im2col_gpu_kernel<<<(channels_col*height_col*width_col+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(data_im,
    		channels, height, width,
    		ksize, stride, pad, dilation, data_col, height_col, width_col, channels_col);
//        dim3 grid((ksize+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS,
//        		(ksize+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS,
//        		(channels+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
//        dim3 block((height_col+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS,
//        		(width_col+CUDA_NUM_THREADS-1)/CUDA_NUM_THREADS);
//        im2col_gpu_kernel_new<<<grid, block>>>(data_im,
//            		channels, height, width,
//            		ksize, stride, pad, dilation, data_col, height_col, width_col, channels_col);

}
