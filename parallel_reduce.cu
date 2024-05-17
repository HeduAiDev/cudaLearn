#include<iostream>
#include<cuda_runtime.h>
#include<nvtx3/nvToolsExt.h>

using namespace std;
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

#define TIMEANDCHECK(kernel,grid,block,args)\
{\
  float milliseconds = 0;\
  CHECK(cudaEventCreate(&start));\
  CHECK(cudaEventCreate(&stop));\
  CHECK(cudaEventRecord(start,0));\
  kernel<<<grid,block>>>args;\
  CHECK(cudaEventRecord(stop,0));\
  CHECK(cudaEventSynchronize(stop));\
  CHECK(cudaEventElapsedTime(&milliseconds,start,stop));\
  CHECK(cudaEventDestroy(start));\
  CHECK(cudaEventDestroy(stop));\
  cudaMemcpy(h_c, d_c, grid*sizeof(float), cudaMemcpyDeviceToHost);\
  CHECK(cudaGetLastError());\
  float ret=0;\
  for(int i = 0; i< grid; i++) {\
       ret += h_c[i];\
  }\
  printf("%-50s: %.6f ms | result: %d(ground truth) == %d  [%s]\n", #kernel,milliseconds, (int)n, (int)ret,  n==ret ? "equal" : "not equal");\
}

__global__ void reduce_sum_baseline(float *c, float *a, long long n) {
    int sum = 0;
    for(int i = 0; i<n; i++) {
        sum += a[i];
    }
    c[0] = sum;
}

__global__ void reduce_sum_normal(float *c, float *a, long long n) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int stride=1; stride < blockDim.x; stride <<= 1) {
        if (tid % (2*stride) == 0) {
            a[gtid] += a[gtid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        c[blockIdx.x] = a[gtid];
    }
}

// 使用shared memory
template <int BlockSize>
__global__ void reduce_sum_shared(float *c, float *a, long long n) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * BlockSize + threadIdx.x;
    __shared__ float smem[BlockSize];
    smem[tid] = a[gtid];
    __syncthreads();
    for(int stride=1; stride < BlockSize; stride <<= 1) {
        // 用位运算代替取模%
        // if ((tid & (2*stride -1))  {
        if (tid % (2*stride) == 0) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        c[blockIdx.x] = smem[0];
    }
}

// 使用shared memory 用位运算代替取模%
template <int BlockSize>
__global__ void reduce_sum_shared_bit(float *c, float *a, long long n) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * BlockSize + threadIdx.x;
    __shared__ float smem[BlockSize];
    smem[tid] = a[gtid];
    __syncthreads();
    for(int stride=1; stride < BlockSize; stride <<= 1) {
        if ((tid & (2*stride -1)) == 0)  {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        c[blockIdx.x] = smem[0];
    }
}

// 避免线程束分化
template <int BlockSize>
__global__ void reduce_sum_shared_wrap_divergence(float *c, float *a, long long n) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * BlockSize + threadIdx.x;
    __shared__ float smem[BlockSize];
    smem[tid] = a[gtid];
    __syncthreads();
    for(unsigned int stride=1; stride < BlockSize; stride <<= 1) {
        int idx = 2*stride*tid;
        if (idx < BlockSize) {
            smem[idx] += smem[idx + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        c[blockIdx.x] = smem[0];
    }
}

// 防止shared memory bank conflict
template <int BlockSize>
__global__ void reduce_sum_shared_bank_conflict(float *c, float *a, long long n) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * BlockSize + threadIdx.x;
    __shared__ float smem[BlockSize];
    smem[tid] = a[gtid];
    __syncthreads();
    for(unsigned int stride=BlockSize/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        c[blockIdx.x] = smem[0];
    }
}

// block管理更多数据, 支持任意大小的block
template <int BlockSize>
__global__ void reduce_sum_shared_bc_stride(float *c, float *a, long long n) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * BlockSize + threadIdx.x;
    __shared__ float smem[BlockSize];
    int total_threads = gridDim.x * BlockSize;
    int sum = 0;
    for(int i = gtid; i < n; i += total_threads) {
        sum += a[i];
    }
    smem[tid] = sum;
    // printf("sum: %d\n", sum);
    __syncthreads();
    for(unsigned int stride=BlockSize/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        c[blockIdx.x] = smem[0];
    }
}

// 循环展开
template <int BlockSize>
__global__ void reduce_sum_shared_bc_stride_expand(float *c, float *a, long long n) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * BlockSize + threadIdx.x;
    __shared__ float smem[BlockSize];
    int total_threads = gridDim.x * BlockSize;
    int sum = 0;
    for(int i = gtid; i < n; i += total_threads) {
        sum += a[i];
    }
    smem[tid] = sum;
    __syncthreads();
    for(unsigned int stride=BlockSize/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32) {
        smem[tid] += smem[tid + 32];
        __syncwarp();
        // if (tid < 16) {
            smem[tid] += smem[tid + 16];
        // }
        __syncwarp();
        // if (tid < 8) {
            smem[tid] += smem[tid + 8];
        // }
        __syncwarp();
        // if (tid < 4) {
            smem[tid] += smem[tid + 4];
        // }
        __syncwarp();
        // if (tid < 2) {
            smem[tid] += smem[tid + 2];
        // }
        __syncwarp();
        // if (tid < 1) {
            smem[tid] += smem[tid + 1];
        // }
        __syncwarp();
    }
    if (tid == 0) {
        c[blockIdx.x] = smem[0];
    }
}

// warp shuffle
template <int BlockSize>
__global__ void reduce_sum_shared_stride_warp_suffle(float *c, float *a, long long n) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * BlockSize + threadIdx.x;
    __shared__ float smem[BlockSize / 32];
    int total_threads = gridDim.x * BlockSize;
    int sum = 0;
    for(int i = gtid; i < n; i += total_threads) {
        sum += a[i];
    }
    int laneid = tid % 32;
    int warpid = tid / 32;
    // {
    //     sum += __shfl_down_sync(0xffffffff, sum, 16);
    //     sum += __shfl_down_sync(0xffffffff, sum, 8);
    //     sum += __shfl_down_sync(0xffffffff, sum, 4);
    //     sum += __shfl_down_sync(0xffffffff, sum, 2);
    //     sum += __shfl_down_sync(0xffffffff, sum, 1);
    // }
    #pragma unroll
    for(int s = 16; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, s);
    }
    if (laneid == 0) {
        smem[warpid] = sum;
    }
    __syncthreads();
    sum = tid < BlockSize / 32 ? smem[tid] : 0;
    if (warpid == 0) {
        // sum += __shfl_down_sync(0xffffffff, sum, 16);
        // sum += __shfl_down_sync(0xffffffff, sum, 8);
        // sum += __shfl_down_sync(0xffffffff, sum, 4);
        // sum += __shfl_down_sync(0xffffffff, sum, 2);
        // sum += __shfl_down_sync(0xffffffff, sum, 1);
        #pragma unroll
        for(int s = 16; s > 0; s >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, s);
        }
    }
    if (tid == 0) {
        c[blockIdx.x] = sum;
    }
}



#define blockSize 512

int main() {
    long long n = 1<<25;
    float *h_a, *d_a;
    float *h_c, *d_c;
    
    cudaEvent_t start, stop;
    int grid = (n-1)/blockSize + 1;
    int block = blockSize;
    h_a = (float*)malloc(n*sizeof(float));
    h_c = (float*)malloc(grid*sizeof(float));
    cudaMalloc((float**)&d_a, n*sizeof(float));
    cudaMalloc((float**)&d_c, grid*sizeof(float));
    for(int i=0;i < n; i++ ) {
        h_a[i] = 1.f;
    }
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    // warmup
    // TIMEANDCHECK(reduce_sum_normal,grid,block,(d_c, d_a, n));
    // cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    // TIMEANDCHECK(reduce_sum_baseline,grid, block,(d_c, d_a, n)); // timeout
    //==================================================
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    TIMEANDCHECK(reduce_sum_normal,grid,block,(d_c, d_a, n));
    //==================================================
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    TIMEANDCHECK(reduce_sum_shared<blockSize>,grid, block,(d_c, d_a, n));
    //==================================================
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    TIMEANDCHECK(reduce_sum_shared_bit<blockSize>,grid, block,(d_c, d_a, n));
    //==================================================
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    TIMEANDCHECK(reduce_sum_shared_wrap_divergence<blockSize>,grid, block,(d_c, d_a, n));
    //==================================================
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    TIMEANDCHECK(reduce_sum_shared_bank_conflict<blockSize>,grid, block,(d_c, d_a, n));
    //==================================================
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    TIMEANDCHECK(reduce_sum_shared_bc_stride<blockSize/2>,grid/2, block/2,(d_c, d_a, n));
    //==================================================
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    TIMEANDCHECK(reduce_sum_shared_bc_stride_expand<blockSize/2>,grid/2, block/2,(d_c, d_a, n));
    //==================================================
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    TIMEANDCHECK(reduce_sum_shared_stride_warp_suffle<blockSize/2>,grid/2, block/2,(d_c, d_a, n));

    free(h_a);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_c);
    return 0;

}