#include<iostream>
#include<cuda_runtime.h>
#include<cute/tensor.hpp>
#include <corecrt_math_defines.h>
#include<cuda_fp16.h>
#include<nvtx3/nvToolsExt.h>
using namespace cute;

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      fprintf(stderr,"ERROR: %s:%d,",__FILE__,__LINE__);\
      fprintf(stderr,"code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}


template<typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
    T val[Size];
    __host__ __device__ __forceinline__ T& operator[](int i) {
        return val[i];
    }
    __host__ __device__ __forceinline__ const T& operator[](int i) const {
        return val[i];
    }
};

template<typename T1, typename T2>
__global__ void check_kernel(T1 *a, T2 *b, bool* flg, unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if(abs((float)a[i] - (float)b[i]) > 1e-3) {
            // printf("a[%d] = %.10f, b[%d] = %.10f\n", i, __half2float(a[i]), i, __half2float(b[i]));
            *flg = false;
            return;
        }
    }
}

template<typename T>
__device__ __host__ __forceinline__ T gelu(T x) {
    T alpha = static_cast<T>(0.7978845608028654);
    T beta = static_cast<T>(0.044714998453855515);
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T tanh_in = alpha * (x + beta * x * x * x);
    const T tanh_out = tanh(tanh_in);
    return half * x * (one + tanh_out);
}

template<>
__device__ __host__ __forceinline__ half2 gelu(half2 x) {
    half2 alpha = __float2half2_rn(0.7978845608028654);
    half2 beta = __float2half2_rn(0.044714998453855515);
    const half2 half = __float2half2_rn(0.5f);
    const half2 one = __float2half2_rn(1.f);
    // const half2 tanh_in = alpha * (x + beta * x * x * x);
    half2 tanh_in_out = __hmul2(alpha, __hadd2(x, __hmul2(beta,__hmul2(x, __hmul2(x, x)))));
    tanh_in_out.x = tanhf(tanh_in_out.x);
    tanh_in_out.y = tanhf(tanh_in_out.y);
    // return half * x * (one + tanh(tanh_in));
    return __hmul2(half, __hmul2(x,__hadd2(one, tanh_in_out)));
}

template<typename T = float>
void print_array(T *a, unsigned int n) {
    for(int i = 0; i < n; i++) {
        printf("%.10f ", a[i]);
    }
    printf("\n");
}

template<>
void print_array(half *a, unsigned int n) {
    for(int i = 0; i < n; i++) {
        printf("%.10f ", __half2float(a[i]));
    }
    printf("\n");
}

template<int grid, int block, typename T1 = float, typename T2 = float>
void check(T1 *a, T2 *b, unsigned int n, std::string suffix = "") {
    bool h_is_equal = true;
    bool* d_is_equal;
    cudaMalloc((void**)&d_is_equal, sizeof(bool));
    cudaMemcpy(d_is_equal, &h_is_equal, sizeof(bool), cudaMemcpyHostToDevice);
    check_kernel<T1,T2><<<grid, block>>>(a, b, d_is_equal, n);
    cudaMemcpy(&h_is_equal, d_is_equal, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("%s is equal: %s\n", suffix.c_str(), h_is_equal == true ? "true" : "false");
}

__global__ void gelu_base_kernel(float* out, float *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        out[i] = gelu<float>(in[i]);
    }
}

__global__ void gelu_vector2_kernel(float* out, float *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int remain = n % 2;
    #pragma unroll
    for(int i = tid; i < n/2; i += blockDim.x * gridDim.x) {
        float2 tmp = reinterpret_cast<float2*>(in)[i];
        tmp.x = gelu<float>(tmp.x);
        tmp.y = gelu<float>(tmp.y);
        reinterpret_cast<float2*>(out)[i] = tmp;
    }
    if ( tid < remain) {
        out[n-1-tid] = gelu<float>(in[n-1-tid]);
    }
}

__global__ void gelu_vector4_kernel(float* out, float *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int remain = n % 4;
    #pragma unroll
    for(int i = tid; i < n/4; i += blockDim.x * gridDim.x) {
        float4 tmp = reinterpret_cast<float4*>(in)[i];
        tmp.x = gelu<float>(tmp.x);
        tmp.y = gelu<float>(tmp.y);
        tmp.z = gelu<float>(tmp.z);
        tmp.w = gelu<float>(tmp.w);
        reinterpret_cast<float4*>(out)[i] = tmp;
    }
    if ( tid < remain) {
        out[n-1-tid] = gelu<float>(in[n-1-tid]);
    }
}

__global__ void gelu_half2_kernel(half* out, half *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int remain = n % 2;
    #pragma unroll
    for(int i = tid; i < n/2; i += blockDim.x * gridDim.x) {
        reinterpret_cast<half2*>(out)[i] = gelu<half2>(reinterpret_cast<half2*>(in)[i]);
    }
    if ( tid < remain) {
        out[n-1-tid] = gelu<half>(in[n-1-tid]);
    }
}

__global__ void gelu_vector2_half2_kernel(half* out, half *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int remain = n % 4;
    // using ArrT = AlignedVector<half2, 2>;
    using ArrT = float2;
    #pragma unroll
    for(int i = tid; i < n/4; i += blockDim.x * gridDim.x) {
        ArrT p = reinterpret_cast<ArrT*>(in)[i];
        #pragma unroll
        for(int j = 0; j < 2; j++) {
            reinterpret_cast<half2*>(&p)[j] = gelu<half2>(reinterpret_cast<half2*>(&p)[j]);
        }
        reinterpret_cast<ArrT*>(out)[i] = p;
    }
    if ( tid < remain) {
        out[n-1-tid] = gelu<half>(in[n-1-tid]);
    }
}

__global__ void gelu_vector4_half2_kernel(half* out, half *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int remain = n % 8;
    // using ArrT = AlignedVector<half2, 4>;
    using ArrT = float4;
    #pragma unroll
    for(int i = tid; i < n/8; i += blockDim.x * gridDim.x) {
        ArrT p = reinterpret_cast<ArrT*>(in)[i];
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            reinterpret_cast<half2*>(&p)[j] = gelu<half2>(reinterpret_cast<half2*>(&p)[j]);
        }
        reinterpret_cast<ArrT*>(out)[i] = p;
    }
    if ( tid < remain) {
        out[n-1-tid] = gelu<half>(in[n-1-tid]);
    }
}

__global__ void gelu_tensor_float_kernel(float* out, float *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < (n+3)/4; i += blockDim.x * gridDim.x) {
        Tensor tin = make_tensor(make_gmem_ptr(in), make_shape(n));
        Tensor tout = make_tensor(make_gmem_ptr(out), make_shape(n));
        Tensor tinr = local_tile(tin, make_shape(Int<8>{}), make_coord(i));
        Tensor toutr = local_tile(tout, make_shape(Int<8>{}), make_coord(i));
        Tensor tinR = make_tensor_like(tinr);
        copy(tinr, tinR);
        #pragma unroll
        for(int x = 0; x < size(tinR); x++) {
            tinR(x) = gelu<float>(tinR(x));
        }
        copy(tinR, toutr);
    }
}

__global__ void gelu_tensor_half2_kernel(half* out, half *in , unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < (n+7)/8; i += blockDim.x * gridDim.x) {
        Tensor tin = make_tensor(make_gmem_ptr(in), make_shape(n));
        Tensor tout = make_tensor(make_gmem_ptr(out), make_shape(n));
        Tensor tinr = local_tile(tin, make_shape(Int<8>{}), make_coord(i));
        Tensor toutr = local_tile(tout, make_shape(Int<8>{}), make_coord(i));
        Tensor tinR = make_tensor_like(tinr);
        copy(tinr, tinR);
        auto tinR2 = recast<half2>(tinR);
        #pragma unroll
        for(int x = 0; x < size(tinR2); x++) {
            tinR2(x) = gelu<half2>(tinR2(x));
        }
        auto tinRx = recast<half>(tinR2);
        copy(tinRx, toutr);
    }
}

void gelu_cpu(float* out, float *in , unsigned int n) {
    for(int i = 0; i < n; i++) {
        out[i] = gelu<float>(in[i]);
    }
}


#define N ((1 << 24))
#define Block 128
#define Grid (((N-1)/(Block*32) + 1) > 65535 ? 65535 : ((N-1)/(Block*32) + 1))

int main() {
    float *h_in, *h_out;
    float *d_in, *d_out, *ground_truth;
    half *h_in_half, *h_out_half;
    half *d_in_half, *d_out_half;
    int size = N * sizeof(int);
    int size_h = N * sizeof(half);
    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);
    h_in_half = (half*)malloc(size_h);
    h_out_half = (half*)malloc(size_h);
    for(int i = 0; i < N; i++) {
        h_in[i] = h_in_half[i] = (float)rand() / (float)RAND_MAX;
    }
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMalloc((void**)&d_in_half, size_h);
    cudaMalloc((void**)&d_out_half, size_h);
    cudaMallocManaged((void**)&ground_truth, size);

    nvtxRangePushA("gelu cpu");
    gelu_cpu(ground_truth, h_in, N);
    nvtxRangePop();
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_half, h_in_half, size_h, cudaMemcpyHostToDevice);

    cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice);
    nvtxRangePushA("gelu base");
    gelu_base_kernel<<<Grid, Block>>>(d_out, d_in, N);
    nvtxRangePop();
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    check<Grid,Block>(ground_truth, d_out, N, "gelu_base");

    cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice);
    gelu_vector2_kernel<<<Grid, Block>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    check<Grid,Block>(ground_truth, d_out, N, "gelu_vector2");

    cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice);
    gelu_vector4_kernel<<<Grid, Block>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    check<Grid,Block>(ground_truth, d_out, N, "gelu_vector4");
    

    gelu_half2_kernel<<<Grid, Block>>>(d_out_half, d_in_half, N);
    cudaMemcpy(h_out_half, d_out_half, size_h, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    check<Grid,Block,float,half>(ground_truth, d_out_half, N, "gelu_half2");

    cudaMemcpy(d_out_half, d_in_half, size_h, cudaMemcpyDeviceToDevice);
    gelu_vector2_half2_kernel<<<Grid, Block>>>(d_out_half, d_in_half, N);
    cudaMemcpy(h_out_half, d_out_half, size_h, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    check<Grid,Block,float,half>(ground_truth, d_out_half, N, "gelu_vector2_half2");

    cudaMemcpy(d_out_half, d_in_half, size_h, cudaMemcpyDeviceToDevice);
    gelu_vector4_half2_kernel<<<Grid, Block>>>(d_out_half, d_in_half, N);
    cudaMemcpy(h_out_half, d_out_half, size_h, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    check<Grid,Block,float,half>(ground_truth, d_out_half, N, "gelu_vector4_half2");

    cudaMemcpy(d_out_half, d_in_half, size_h, cudaMemcpyDeviceToDevice);
    gelu_tensor_half2_kernel<<<Grid, Block>>>(d_out_half, d_in_half, N);
    cudaMemcpy(h_out_half, d_out_half, size_h, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    check<Grid,Block,float,half>(ground_truth, d_out_half, N, "gelu_tensor_half2_kernel");

    cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice);
    gelu_tensor_float_kernel<<<Grid, Block>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    check<Grid,Block>(ground_truth, d_out, N, "gelu_tensor_float_kernel");

    // printf("int:\n");
    // print_array(h_in, N);
    // printf("out:\n");
    // print_array(h_out, N);
    // printf("out_half:\n");
    // print_array<half>(h_out_half, N);
    // printf("ground_truth:\n");
    // print_array(ground_truth, N);
    
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}