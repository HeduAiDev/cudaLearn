#include<iostream>

void filter_cpu(int *out, unsigned int *res, int *src, int n) {
    for (int i = 0; i < n; i++) {
        if (src[i] == 1) {
            out[(*res)++] = src[i];
        }
    }
}

__global__ void filter_global(int *out, unsigned int *res, int *src, int n) {
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    for(int i = gtid; i < n; i += total_threads) {
        if (src[i] == 1) {
            out[atomicAdd(res, 1)] = src[i];
        }
    }
}

__global__ void filter_shared(int *out, unsigned int *res, int *src, int n) {
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    __shared__ unsigned int cnt;
    __shared__ unsigned int offset_g;
    unsigned int offset_s;
    for(int i = gtid; i < n; i+=total_threads) {
        if (tid == 0) {
            cnt = 0;
        }
        __syncthreads();
        if (src[i] == 1) {
            offset_s = atomicAdd(&cnt, 1);
        }
        __syncthreads();
        if (tid == 0) {
            offset_g = atomicAdd(res, cnt);
        }
        __syncthreads();
        if (src[i] == 1) {
            out[offset_g + offset_s] = src[i];
        }
    }
}

__global__ void filter_global_warp_aggregated(int *out, unsigned int *res, int *src, int n) {
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneid = threadIdx.x % 32;
    int total_threads = gridDim.x * blockDim.x;
    for(int i = gtid; i < n; i += total_threads) {
        if (src[i] == 1) {
            int active = __activemask();
            int cnt = __popc(active);
            int rank = __popc(active & ((1<<laneid) - 1));
            int offset_g;
            if (rank == 0) {
                offset_g = atomicAdd(res, cnt);
            }
            offset_g = __shfl_sync(active, offset_g, 0);
            out[offset_g + rank] = src[i];
        }
    }
}

__global__ void filter_shared_warp_aggregated(int *out, unsigned int *res, int *src, int n) {
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int laneid = tid % 32;
    int total_threads = gridDim.x * blockDim.x;
    __shared__ unsigned int smem_cnt;
    __shared__ unsigned int offset_g;
    unsigned int offset_s;
    for(int i = gtid; i < n; i+=total_threads) {
        if (tid == 0) {
            smem_cnt = 0;
        }
        // 同步smem_cnt
        __syncthreads();
        if (src[i] == 1) {
            // 获取一个warp中的增量
            // offset_s 为当前warp在smem中的起始偏移量
            // rank 为所有参与计算的线程的相对id
            int active = __activemask();
            int cnt = __popc(active);
            // 我发现nv博客中提到的__lanemask_lt()函数并不存在，于是手动实现了一个等效的 (1<<laneid) -1
            int rank = __popc(active & ((1<<laneid) - 1));
            // 必须是rank 0而不是laneid 0的原因为，laneid 0的线程可能没有参与计算。
            if (rank == 0) {
                offset_s = atomicAdd(&smem_cnt, cnt);
            }
            offset_s = __shfl_sync(active, offset_s, 0);
        }
        // 同步block中所有warp,以获取一个block的增量
        __syncthreads();
        if (tid == 0) {
            offset_g = atomicAdd(res, smem_cnt);
        }
        // offset_g是smem，因此得同步一下。
        __syncthreads();
        if (src[i] == 1) {
            int rank = __popc(__activemask() & ((1<<laneid) - 1));
            out[offset_g + offset_s + rank] = src[i];
        }
    }
}


int main() {
    // unsigned int t = time(NULL);
    // printf("time: %d\n",t);
    // srand(t);
    int N = 1 << 24;
    unsigned int h_res = 0, *d_res;
    int *h_src, *d_src;
    int *h_out, *d_out;

    int block = 1024;
    int grid = 100;
    
    // Allocate memory on host
    h_src = (int*)malloc(N * sizeof(int));
    h_out = (int*)malloc(N * sizeof(int));
    
    // Initialize input array
    for (int i = 0; i < N; i++) {
        // h_src[i] = i % 2;
        h_src[i] = 1;
    }
    
    // Allocate memory on device
    cudaMalloc((void**)&d_src, N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));
    cudaMalloc((void**)&d_res, sizeof(unsigned int));
    

    // unsigned int ground_truth = N / 2;
    unsigned int ground_truth = N;
    // filter_cpu(h_out, &ground_truth, h_src, N);
    
    // Copy input array to device
    cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // Launch kernel
    filter_global<<<grid, block>>>(d_out, d_res, d_src, N);
    // Copy result from device to host
    cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    // Print result
    printf("filter_global                : %d(ground truth) == %d [%s]\n", ground_truth, h_res, (ground_truth == h_res) ? "equal" : "not equal");

    
    // Copy input array to device
    h_res = 0;
    cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // Launch kernel
    filter_shared<<<grid, block>>>(d_out, d_res, d_src, N);
    // Copy result from device to host
    cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    // Print result
    printf("filter_shared                : %d(ground truth) == %d [%s]\n", ground_truth, h_res, (ground_truth == h_res) ? "equal" : "not equal");

    // Copy input array to device
    h_res = 0;
    cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // Launch kernel
    filter_global_warp_aggregated<<<grid, block>>>(d_out, d_res, d_src, N);
    // Copy result from device to host
    cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    // Print result
    printf("filter_global_warp_aggregated: %d(ground truth) == %d [%s]\n", ground_truth, h_res, (ground_truth == h_res) ? "equal" : "not equal");

    // Copy input array to device
    h_res = 0;
    cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // Launch kernel
    filter_shared_warp_aggregated<<<grid, block>>>(d_out, d_res, d_src, N);
    // Copy result from device to host
    cudaMemcpy(&h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    // Print result
    printf("filter_shared_warp_aggregated: %d(ground truth) == %d [%s]\n", ground_truth, h_res, (ground_truth == h_res) ? "equal" : "not equal");
    
    // Free memory
    cudaFree(d_src);
    cudaFree(d_out);
    cudaFree(d_res);
    free(h_src);
    free(h_out);
}