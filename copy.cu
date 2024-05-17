#include<iostream>
#include<cuda_runtime.h>

__global__ void check_kernel(float *a, float *b, bool* flg, unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if(a[i] != b[i]) {
            *flg = false;
        }
    }
}

template<int grid, int block>
void check(float *a, float *b, unsigned int n, std::string suffix = "") {
    bool h_is_equal = true;
    bool* d_is_equal;
    cudaMalloc((void**)&d_is_equal, sizeof(bool));
    cudaMemcpy(d_is_equal, &h_is_equal, sizeof(bool), cudaMemcpyHostToDevice);
    check_kernel<<<grid, block>>>(a, b, d_is_equal, n);
    cudaMemcpy(&h_is_equal, d_is_equal, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("%s is equal: %s\n", suffix.c_str(), h_is_equal == true ? "true" : "false");
}

__global__ void copy(float*out, float *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        out[i] = in[i];
    }
}

__global__ void copy_vector2(float*out, float *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n/2; i += blockDim.x * gridDim.x) {
        reinterpret_cast<int2*>(out)[i] = reinterpret_cast<const int2*>(in)[i];
    }
}

__global__ void copy_vector4(float*out, float *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n/4; i += blockDim.x * gridDim.x) {
        reinterpret_cast<int4*>(out)[i] = reinterpret_cast<const int4*>(in)[i];
    }
}
#define N (1 << 25)
#define Block 128
#define Grid (((N-1)/(Block*32) + 1) > 65535 ? 65535 : ((N-1)/(Block*32) + 1))

int main() {
    float *h_in, *h_out;
    float *d_in, *d_out;
    int size = N * sizeof(int);
    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);
    for(int i = 0; i < N; i++) {
        h_in[i] = i;
    }
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    copy<<<Grid, Block>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    check<Grid,Block>(d_in, d_out, N, "copy");

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    copy_vector2<<<Grid, Block>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    check<Grid,Block>(d_in, d_out, N, "copy_vector2");


    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    copy_vector4<<<Grid, Block>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    check<Grid,Block>(d_in, d_out, N, "copy_vector4");
    
    std::cout <<"in:"<< std::endl;
    for(int i = 0; i < 20; i++) {
        std::cout << h_in[i] << " ";
    }
    std::cout << std::endl;
    std::cout <<"out:"<< std::endl;
    for(int i = 0; i < 20; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}