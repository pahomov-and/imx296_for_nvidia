//
// Created by Andrey Pahomov on 28.08.20.
//

#include <cuda_runtime.h>
#include "histogram.cuh"


__device__ inline float clamp(float val, float mn, float mx) {
    return (val >= mn) ? ((val <= mx) ? val : mx) : mn;
}

__global__ void gpuConvertY10to8uc1_kernel(unsigned short *src, unsigned char *dst,
                                           unsigned int width, unsigned int height) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= width || row >= height)
        return;

    __syncthreads();

    dst[row * width + col] = src[row * width + col] >> 2;
}


__global__ void gpuConvertY10to8uc1_kernel_gist_simple(unsigned short *src, unsigned char *dst,
                                                       unsigned int width, unsigned int height,
                                                       unsigned int *intensity_num,
                                                       double *intensity_pro,
                                                       unsigned int *min_index,
                                                       unsigned int *max_index) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pos = row * width + col;

    if (col >= width || row >= height)
        return;

    /**********************count_intensity********************/
    atomicAdd(&intensity_num[(unsigned int) (src[pos] & INTENSITY_MASK)], 1);
    /******************************************/
    __syncthreads();

    if (*max_index > *min_index && *max_index > 0) {
        double k = (double) (255. / (*max_index - *min_index));
        dst[pos] = (unsigned char) ((double) (src[pos] - *min_index) * k);
    } else dst[pos] = 0;

}

__global__ void gpuConvertY10to8uc1_kernel_gist_optimized(unsigned short *src, unsigned char *dst,
                                                          unsigned int width, unsigned int height,
                                                          unsigned int *intensity_num,
                                                          double *intensity_pro,
                                                          unsigned int *min_index,
                                                          unsigned int *max_index) {

    unsigned int stride;
    __shared__ unsigned int shared_bin[INTENSITY_RANGE];
    unsigned long long i;

    shared_bin[threadIdx.x] = 0;
    __syncthreads();

    i = blockIdx.x * blockDim.x + threadIdx.x;
    stride = blockDim.x * gridDim.x;

    if (i > width * height) return;

    while (i < width * height) {
        atomicAdd(&shared_bin[src[i]], 1);
        i += stride;
    }

    __syncthreads();
    if (threadIdx.x < INTENSITY_RANGE)
        atomicAdd(&intensity_num[threadIdx.x], shared_bin[threadIdx.x]);


    if (threadIdx.x == 0) {
        unsigned int offset = 1;
        unsigned int offset_l = 1;
        unsigned int offset_r = 1;
        bool isMin = false;
        bool isMax = false;

        for (int i = offset_l; i < INTENSITY_RANGE; i++) {

            if (intensity_num[i] > offset && !isMin) {
                *min_index = i;
                isMin = true;
            }

            if (intensity_num[INTENSITY_RANGE - i - offset_r] > offset && !isMax) {
                *max_index = INTENSITY_RANGE - i - offset_r;
                isMax = true;
            }

            if (isMin && isMax) break;

        }

    }

}


/*dynamic range expansion*/
__global__ void gpuConvertY10to8uc1_kernel_dre(unsigned short *src, unsigned char *dst,
                                               unsigned int width, unsigned int height,
                                               unsigned int *intensity_num,
                                               double *intensity_pro,
                                               unsigned int *min_index,
                                               unsigned int *max_index) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pos = row * width + col;

    __syncthreads();

    if (col >= width || row >= height)
        return;


    if (*max_index > *min_index && *max_index > 0) {

        const unsigned int diff = *max_index - *min_index;
        dst[pos] = (unsigned char) ((src[pos] - *min_index) * 255 / diff);

    } else dst[pos] = 0;

}

void gpuConvertY10to8uc1(unsigned short *src, unsigned char *dst,
                         unsigned int width, unsigned int height) {
    unsigned short *d_src = NULL;
    unsigned char *d_dst = NULL;
    size_t planeSize = width * height * sizeof(unsigned char);

    unsigned int flags;
    bool srcIsMapped = (cudaHostGetFlags(&flags, src) == cudaSuccess) && (flags & cudaHostAllocMapped);
    bool dstIsMapped = (cudaHostGetFlags(&flags, dst) == cudaSuccess) && (flags & cudaHostAllocMapped);

    if (srcIsMapped) {
        d_src = src;
        cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);
    } else {
        cudaMalloc(&d_src, planeSize * 2);
        cudaMemcpy(d_src, src, planeSize * 2, cudaMemcpyHostToDevice);
    }
    if (dstIsMapped) {
        d_dst = dst;
        cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);
    } else {
        cudaMalloc(&d_dst, planeSize * 3);
    }


    int threadNum = 32;
    dim3 blockSize = dim3(threadNum, threadNum, 1);
    dim3 gridSize = dim3(width / threadNum + 1, height / threadNum + 1, 1);
    gpuConvertY10to8uc1_kernel << < gridSize, blockSize >> > (d_src, d_dst, width, height);

//    unsigned int blockSize = 1024;
//    unsigned int numBlocks = (width / 2 + blockSize - 1) / blockSize;
//    gpuConvertY10to8uc1_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, width, height);

//    cudaDeviceSynchronize();
    cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);
    cudaStreamSynchronize(NULL);

    if (!srcIsMapped) {
        cudaMemcpy(dst, d_dst, planeSize * 3, cudaMemcpyDeviceToHost);
        cudaFree(d_src);
    }
    if (!dstIsMapped) {
        cudaFree(d_dst);
    }
}

void gpuConvertY10to8uc1_gist(unsigned short *src, unsigned char *dst,
                              unsigned int width, unsigned int height,
                              unsigned int *intensity_num,
                              double *intensity_pro,
                              unsigned int *min_index,
                              unsigned int *max_index,
                              bool isMatGist) {
    unsigned short *d_src = NULL;
    unsigned char *d_dst = NULL;
    unsigned int *d_intensity_num = NULL;
    double *d_intensity_pro = NULL;
    unsigned int *d_min_index = NULL;
    unsigned int *d_max_index = NULL;

    size_t planeSize = width * height * sizeof(unsigned char);

    unsigned int flags;
    bool srcIsMapped = (cudaHostGetFlags(&flags, src) == cudaSuccess) && (flags & cudaHostAllocMapped);
    bool dstIsMapped = (cudaHostGetFlags(&flags, dst) == cudaSuccess) && (flags & cudaHostAllocMapped);
    bool intensity_numIsMapped =
            (cudaHostGetFlags(&flags, intensity_num) == cudaSuccess) && (flags & cudaHostAllocMapped);
    bool intensity_proIsMapped =
            (cudaHostGetFlags(&flags, intensity_pro) == cudaSuccess) && (flags & cudaHostAllocMapped);
    bool min_indexIsMapped = (cudaHostGetFlags(&flags, min_index) == cudaSuccess) && (flags & cudaHostAllocMapped);
    bool max_indexIsMapped = (cudaHostGetFlags(&flags, max_index) == cudaSuccess) && (flags & cudaHostAllocMapped);

    if (srcIsMapped) {
        d_src = src;
        cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);
    } else {
        cudaMalloc(&d_src, planeSize * 2);
        cudaMemcpy(d_src, src, planeSize * 2, cudaMemcpyHostToDevice);
    }
    if (dstIsMapped) {
        d_dst = dst;
        cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);
    } else {
        cudaMalloc(&d_dst, planeSize * 3);
    }

    if (intensity_numIsMapped) {
        d_intensity_num = intensity_num;
        cudaStreamAttachMemAsync(NULL, intensity_num, 0, cudaMemAttachGlobal);
    } else {
        cudaMalloc(&d_intensity_num, INTENSITY_RANGE * sizeof(unsigned int));
        cudaMemcpy(d_intensity_num, intensity_num, INTENSITY_RANGE * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    if (intensity_proIsMapped) {
        d_intensity_pro = intensity_pro;
        cudaStreamAttachMemAsync(NULL, intensity_pro, 0, cudaMemAttachGlobal);
    } else {
        cudaMalloc(&d_intensity_pro, INTENSITY_RANGE * sizeof(double));
        cudaMemcpy(d_intensity_pro, intensity_pro, INTENSITY_RANGE * sizeof(double), cudaMemcpyHostToDevice);
    }

    if (min_indexIsMapped) {
        d_min_index = min_index;
        cudaStreamAttachMemAsync(NULL, min_index, 0, cudaMemAttachGlobal);
    } else {
        cudaMalloc(&d_min_index, 1 * sizeof(unsigned int));
        cudaMemcpy(d_min_index, min_index, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    if (max_indexIsMapped) {
        d_max_index = max_index;
        cudaStreamAttachMemAsync(NULL, max_index, 0, cudaMemAttachGlobal);
    } else {
        cudaMalloc(&d_max_index, 1 * sizeof(unsigned int));
        cudaMemcpy(d_max_index, max_index, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }


    int threadNum = INTENSITY_RANGE;
    dim3 blockSize(INTENSITY_RANGE);
    dim3 gridSize = dim3(width / threadNum + 1, height / threadNum + 1, 1);

    if (isMatGist)
        gpuConvertY10to8uc1_kernel_gist_optimized << < gridSize, blockSize >> >
                                                                 (d_src, d_dst, width, height,
                                                                         d_intensity_num,
                                                                         d_intensity_pro,
                                                                         d_min_index, d_max_index);


    threadNum = 32;
    blockSize = dim3(threadNum, threadNum, 1);
    gridSize = dim3(width / threadNum + 1, height / threadNum + 1, 1);

    gpuConvertY10to8uc1_kernel_dre << < gridSize, blockSize >> >
                                                  (d_src, d_dst, width, height,
                                                          d_intensity_num,
                                                          d_intensity_pro,
                                                          d_min_index, d_max_index);

    cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(NULL, intensity_num, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(NULL, intensity_pro, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(NULL, min_index, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(NULL, max_index, 0, cudaMemAttachHost);
    cudaStreamSynchronize(NULL);

    if (!srcIsMapped) {
        cudaMemcpy(dst, d_dst, planeSize * 3, cudaMemcpyDeviceToHost);
        cudaFree(d_src);
    }
    if (!dstIsMapped) {
        cudaFree(d_dst);
    }

    if (!intensity_numIsMapped) {
        cudaFree(d_intensity_num);
    }

    if (!intensity_proIsMapped) {
        cudaFree(d_intensity_pro);
    }

    if (!min_indexIsMapped) {
        cudaFree(d_min_index);
    }

    if (!min_indexIsMapped) {
        cudaFree(d_max_index);
    }
}
