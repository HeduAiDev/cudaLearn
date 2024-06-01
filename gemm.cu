//#ncu-add
#include <iostream>
#include <cute/tensor.hpp>
#include <cublas_v2.h>

using namespace cute;

#define M 1024
#define N 65536
#define K 1024

#define TileM 128
#define TileN 128
#define TileK 8

#define KTileM  (((M) + (TileM - 1))/TileM)
#define KTileK  (((K) + (TileK - 1))/TileK) 
#define KTileN  (((N) + (TileN - 1))/TileN)

#define REGM 8
#define REGN 8


#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      fprintf(stderr,"ERROR: %s:%d,",__FILE__,__LINE__);\
      fprintf(stderr,"code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      throw std::exception("cuda error");\
  }\
}


template<class T>
void gen_rand_data(T *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] =(int)(rand() % 10 - 5);
    }
}

template<class T>
__host__ __device__ __forceinline__ void print_matrix(T *a, int rows, int cols, char prefix[] = "") 
{
    printf("Matrix %s:\n", prefix);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%5.0f", __half2float(a[i * cols + j]));
        }
        printf("\n");
    }
}
// bygpt
void print_centered(const char *str, int width, char fill_char) {
    int len = strlen(str);
    if (len >= width) {
        // 如果字符串长度大于等于指定宽度，直接输出字符串
        printf("%s\n", str);
    } else {
        // 计算左右填充的长度
        int total_padding = width - len;
        int left_padding = total_padding / 2;
        int right_padding = total_padding - left_padding;

        // 输出左填充字符
        for (int i = 0; i < left_padding; i++) {
            putchar(fill_char);
        }

        // 输出字符串
        printf("%s", str);

        // 输出右填充字符
        for (int i = 0; i < right_padding; i++) {
            putchar(fill_char);
        }

        // 换行
        putchar('\n');
    }
}

struct Timeit {
    cudaEvent_t e_start;
    cudaEvent_t e_stop;
    float elapsed_time;
     __inline__ void start() {
        CHECK(cudaEventCreate(&e_start));
        CHECK(cudaEventCreate(&e_stop));
        CHECK(cudaEventRecord(e_start, 0));
    }
    __inline__ void stop() {
        CHECK(cudaEventRecord(e_stop,0));
        CHECK(cudaEventSynchronize(e_stop));
        CHECK(cudaEventElapsedTime(&elapsed_time, e_start, e_stop));
        CHECK(cudaEventDestroy(e_start));
        CHECK(cudaEventDestroy(e_stop));
    }
    __inline__ float get_FLOPS()
    {
       return ((float)M * N * K * 2) / (elapsed_time) / 1e6;
    }
};

template<typename T1, typename T2>
__global__ void check_kernel(T1 *a, T2 *b, bool* flg, unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if(abs((float) (a[i] - b[i]) > 1e-6)) {
            // printf("a[%d] = %.10f, b[%d] = %.10f\n", i, __half2float(a[i]), i, __half2float(b[i]));
            *flg = false;
        }
        if (*flg == false) return;
    }
}

template<int grid, int block, typename T1 = half, typename T2 = half>
void check(T1 *h_a, T2 *h_b, unsigned int n, std::string suffix = "") {
    bool h_is_equal = true;
    T1 *d_a;
    T2 *d_b;
    bool* d_is_equal;
    CHECK(cudaMalloc((void**)&d_is_equal, sizeof(bool)));
    CHECK(cudaMemcpy(d_is_equal, &h_is_equal, sizeof(bool), cudaMemcpyHostToDevice));
    cudaMalloc((void**)&d_a, n * sizeof(T1));
    cudaMalloc((void**)&d_b, n * sizeof(T2));
    CHECK(cudaMemcpy(d_a, h_a, n * sizeof(T1), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, n * sizeof(T2), cudaMemcpyHostToDevice));
    check_kernel<T1,T2><<<grid, block>>>(d_a, d_b, d_is_equal, n);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());    
    CHECK(cudaMemcpy(&h_is_equal, d_is_equal, sizeof(bool), cudaMemcpyDeviceToHost));
    printf("%s is equal: %s\n", suffix.c_str(), h_is_equal == true ? "true" : "false");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_is_equal);
}

template<class T>
void matrix_cpu(T *a, T *b, T *c, int m, int n, int k)
{
    for (int y = 0; y < m; y++)
        for (int x = 0; x < n; x++)
        {
            T sum = 0;
            for (int i = 0; i < k; i++)
            {
                sum += a[y * k + i] * b[x * k + i];
            }
            c[y * n + x] = sum;
        }
}

__global__ void gmem_kernel(half *a, half *b, half *c, int m, int n, int k)
{
    int total_threads = gridDim.x * blockDim.x;
    #pragma unroll
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < m; row += total_threads)
    {
        #pragma unroll
        for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < n; col += total_threads) 
        {
            half sum = 0;
            for (int i = 0; i <k; i++)
            {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

// TileM,TileN
__global__ void tile_smem_kernel(half *a, half *b, half *c, int m, int n, int k)
{
    __shared__ half sA[TileM][TileK];
    __shared__ half sB[TileK][TileN];
    int row = blockIdx.x * TileM + threadIdx.x;
    int col = blockIdx.y * TileN + threadIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    half sum = 0;

    #pragma unroll
    for (int tki = 0; tki < KTileK; tki++)
    {
        #pragma unroll
        for(int i = tid; i < TileM * TileK; i += total_threads)        
        {
            int y = i % TileM;
            int x = i / TileM;
            sA[y][x] = a[(blockIdx.x * TileM + y) * k + tki * TileK + x];
        }
        #pragma unroll
        for(int i = tid; i < TileK * TileN; i += total_threads)        
        {
            int y = i % TileK;
            int x = i / TileK;
            sB[y][x] = b[(tki * TileK + y) * n + blockIdx.y * TileN + x];
        }
        __syncthreads();

        // 支持任意形状的 TileK
        #pragma unroll
        for (int i = 0; i < min(k - tki * TileK, TileK); i++)
        {
            sum += sA[threadIdx.x][i] * sB[i][threadIdx.y];
        }
    }
    if (row < m && col < n)
    {
        c[row * n + col] = sum;
    }
}

// TileM, TileN
__global__ void tile_smem_float4_kernel(half *a, half *b, half *c, int m, int n, int k)
{
    __shared__ half sA[TileM][TileK];
    __shared__ half sB[TileK][TileN];
    int row = blockIdx.x * TileM + threadIdx.x;
    int col = blockIdx.y * TileN + threadIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    half sum = 0;

    // sA
    #if (TileK % 8 == 0 && K % TileK == 0)
    using ArrT_a = float4;
    #define elements_a 8
    #elif (TileK % 4 == 0 && K % TileK == 0)
    using ArrT_a = float2;
    #define elements_a 4
    #elif (TileK % 2 == 0 && K % TileK == 0)
    using ArrT_a = float;
    #define elements_a 2
    #else
    using ArrT_a = half;
    #define elements_a 1
    #endif
    // sB
    #if (TileN % 8 == 0 && N % TileN == 0)
    using ArrT_b = float4;
    #define elements_b 8
    #elif (TileN % 4 == 0 && N % TileN == 0)
    using ArrT_b = float2;
    #define elements_b 4
    #elif (TileN % 2 == 0 && N % TileN == 0)
    using ArrT_b = float;
    #define elements_b 2
    #else
    using ArrT_b = half;
    #define elements_b 1
    #endif
   
    #pragma unroll
    for (int tki = 0; tki < KTileK; tki++)
    {
        #pragma unroll
        for(int i = tid; i < TileM * TileK / elements_a; i += total_threads)        
        {
            int y = i % TileM;
            int x = i / TileM;
            reinterpret_cast<ArrT_a*>(&sA[y][x * elements_a])[0] = reinterpret_cast<ArrT_a*>(&a[(blockIdx.x * TileM + y) * k + tki * TileK + x * elements_a])[0];
        }
        #pragma unroll
        for(int i = tid; i < TileK * TileN / elements_b; i += total_threads)        
        {
            int y = i % TileK;
            int x = i / TileK;
            reinterpret_cast<ArrT_b*>(&sB[y][x * elements_b])[0] = reinterpret_cast<ArrT_b*>(&b[(tki * TileK + y) * n + blockIdx.y * TileN + x * elements_b])[0];
        }
        __syncthreads();

        // 允许k 不能被 TileK整除
        #pragma unroll
        for (int i = 0; i < min(k - tki * TileK, TileK); i++)
        {
            sum += sA[threadIdx.x][i] * sB[i][threadIdx.y];
        }
    }
    if (row < m && col < n)
    {
        c[row * n + col] = sum;
    }
}


// TileM/REGM, TileN/REGN
__global__ void tile_smem_float4_tile_reg_kernel(half * __restrict__ a, half * __restrict__ b, half * __restrict__ c, int m, int n, int k)
{
    __shared__ half sA[TileM][TileK];
    __shared__ half sB[TileK][TileN];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    half fragA[REGM];
    half fragB[REGN];
    half fragC[REGM][REGN] = {0};
    if (threadIdx.x >= TileM/REGM || threadIdx.y >= TileN/REGN) return;
    // sA
    #if (TileK % 8 == 0 && K % TileK == 0)
    using ArrT_a = float4;
    #define elements_a 8
    #elif (TileK % 4 == 0 && K % TileK == 0)
    using ArrT_a = float2;
    #define elements_a 4
    #elif (TileK % 2 == 0 && K % TileK == 0)
    using ArrT_a = float;
    #define elements_a 2
    #else
    using ArrT_a = half;
    #define elements_a 1
    #endif
    // sB
    #if (TileN % 8 == 0 && N % TileN == 0)
    using ArrT_b = float4;
    #define elements_b 8
    #elif (TileN % 4 == 0 && N % TileN == 0)
    using ArrT_b = float2;
    #define elements_b 4
    #elif (TileN % 2 == 0 && N % TileN == 0)
    using ArrT_b = float;
    #define elements_b 2
    #else
    using ArrT_b = half;
    #define elements_b 1
    #endif
    // REG
    #if (REGN % 8 == 0)
    using ArrT_c = float4;
    #define elements_c 8
    #elif (REGN % 4 == 0)
    using ArrT_c = float2;
    #define elements_c 4
    #elif (REGN % 2 == 0)
    using ArrT_c = float;
    #define elements_c 2
    #else
    using ArrT_c = half;
    #define elements_c 1
    #endif
    #pragma unroll
    for (int tki = 0; tki < KTileK; tki++)
    {
        #pragma unroll
        for(int i = tid; i < TileM * TileK / elements_a; i += total_threads)        
        {
            int y = i % TileM;
            int x = i / TileM;
            reinterpret_cast<ArrT_a*>(&sA[y][x * elements_a])[0] = reinterpret_cast<ArrT_a*>(&a[(blockIdx.x * TileM + y) * k + tki * TileK + x * elements_a])[0];
        }
        #pragma unroll
        for(int i = tid; i < TileK * TileN / elements_b; i += total_threads)        
        {
            int y = i % TileK;
            int x = i / TileK;
            reinterpret_cast<ArrT_b*>(&sB[y][x * elements_b])[0] = reinterpret_cast<ArrT_b*>(&b[(tki * TileK + y) * n + blockIdx.y * TileN + x * elements_b])[0];
        }
        __syncthreads();

        // 允许k 不能被 TileK整除
        #pragma unroll
        for (int i = 0; i < min(k - tki * TileK, TileK); i++)
        {
            #pragma unroll
            for(int j = 0; j < REGM; j++)    
            {
                fragA[j] = sA[threadIdx.x * REGM + j][i];
            } 
            #pragma unroll
            for (int j = 0; j < REGN / elements_c; j++)
                reinterpret_cast<ArrT_c *>(fragB)[j] = reinterpret_cast<ArrT_c *>(&sB[i][threadIdx.y * REGN])[j];
            #pragma unroll
            for (int y = 0; y < REGM; y++) {
                for (int x = 0; x < REGN; x++)
                {
                    fragC[y][x] += fragA[y] * fragB[x];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int y = 0; y < REGM; y++) 
    {
        #if (REGN % 8 == 0 && N % REGN == 0)
        #pragma unroll
        for (int x = 0; x < REGN / 8; x++)
            reinterpret_cast<float4 *>(&c[(blockIdx.x * TileM + threadIdx.x * REGM + y) * n + blockIdx.y * TileN + threadIdx.y * REGN])[x] = reinterpret_cast<float4 *>(&fragC[y][0])[x];
        #elif (REGN % 4 == 0 && N % REGN == 0)
        #pragma unroll
        for (int x = 0; x < REGN / 4; x++)
            reinterpret_cast<float2 *>(&c[(blockIdx.x * TileM + threadIdx.x * REGM + y) * n + blockIdx.y * TileN + threadIdx.y * REGN])[x] = reinterpret_cast<float2 *>(&fragC[y][0])[x];
        #else
        #pragma unroll
        for(int x = 0; x < REGN && blockIdx.x * TileM + threadIdx.x * REGM + y < M && blockIdx.y * TileN + threadIdx.y * REGN + x < N; x++)
        {
            c[(blockIdx.x * TileM + threadIdx.x * REGM + y) * n + blockIdx.y * TileN + threadIdx.y * REGN + x] = fragC[y][x];
        }
        #endif
    }
}

// TileM/REGM, TileN/REGN
__global__ void tile_smem_float4_tile_reg_BT_kernel(half * __restrict__ a, half * __restrict__ b, half * __restrict__ c, int m, int n, int k)
{
    __shared__ half sA[TileK][TileM];
    __shared__ half sB[TileK][TileN];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    half fragA[REGM];
    half fragB[REGN];
    half fragC[REGM][REGN] = {0};
    if (threadIdx.x >= TileM/REGM || threadIdx.y >= TileN/REGN) return;
    // sA
    #if (TileK % 8 == 0 && K % TileK == 0)
    using ArrT_a = float4;
    #define elements_a 8
    #elif (TileK % 4 == 0 && K % TileK == 0)
    using ArrT_a = float2;
    #define elements_a 4
    #elif (TileK % 2 == 0 && K % TileK == 0)
    using ArrT_a = float;
    #define elements_a 2
    #else
    using ArrT_a = half;
    #define elements_a 1
    #endif
    // sB
    #if (TileN % 8 == 0 && N % TileN == 0)
    using ArrT_b = float4;
    #define elements_b 8
    #elif (TileN % 4 == 0 && N % TileN == 0)
    using ArrT_b = float2;
    #define elements_b 4
    #elif (TileN % 2 == 0 && N % TileN == 0)
    using ArrT_b = float;
    #define elements_b 2
    #else
    using ArrT_b = half;
    #define elements_b 1
    #endif
    // REG
    #if (REGN % 8 == 0)
    using ArrT_cn = float4;
    #define elements_cn 8
    #elif (REGN % 4 == 0)
    using ArrT_cn = float2;
    #define elements_cn 4
    #elif (REGN % 2 == 0)
    using ArrT_cn = float;
    #define elements_cn 2
    #else
    using ArrT_cn = half;
    #define elements_cn 1
    #endif
    #if (REGM % 8 == 0)
    using ArrT_cm = float4;
    #define elements_cm 8
    #elif (REGM % 4 == 0)
    using ArrT_cm = float2;
    #define elements_cm 4
    #elif (REGM % 2 == 0)
    using ArrT_cm = float;
    #define elements_cm 2
    #else
    using ArrT_cm = half;
    #define elements_cm 1
    #endif
    #pragma unroll
    for (int tki = 0; tki < KTileK; tki++)
    {
        #pragma unroll
        for(int i = tid; i < TileM * TileK / elements_a; i += total_threads)        
        {
            half tmp_a[elements_a];
            int y = i % TileM;
            int x = i / TileM;
            reinterpret_cast<ArrT_a*>(tmp_a)[0] = reinterpret_cast<ArrT_a*>(&a[(blockIdx.x * TileM + y) * k + tki * TileK + x * elements_a])[0];
            #pragma unroll
            for(int j = 0; j < elements_a; j++)    
            {
                sA[x * elements_a + j][y] = tmp_a[j];
            }
        }
        #pragma unroll
        for(int i = tid; i < TileK * TileN / elements_b; i += total_threads)        
        {
            int y = i % TileK;
            int x = i / TileK;
            reinterpret_cast<ArrT_b*>(&sB[y][x * elements_b])[0] = reinterpret_cast<ArrT_b*>(&b[(tki * TileK + y) * n + blockIdx.y * TileN + x * elements_b])[0];
            
        }
        __syncthreads();

        // 允许k 不能被 TileK整除
        #pragma unroll
        for (int i = 0; i < min(k - tki * TileK, TileK); i++)
        {
            #pragma unroll
            for (int j = 0; j < REGM / elements_cm; j++)
                reinterpret_cast<ArrT_cm *>(fragA)[j] = reinterpret_cast<ArrT_cm *>(&sA[i][threadIdx.x * REGM])[j];
            #pragma unroll
            for (int j = 0; j < REGN / elements_cn; j++)
                reinterpret_cast<ArrT_cn *>(fragB)[j] = reinterpret_cast<ArrT_cn *>(&sB[i][threadIdx.y * REGN])[j];
            // #pragma unroll
            // for (int y = 0; y < REGM; y++) {
            //     for (int x = 0; x < REGN; x++)
            //     {
            //         fragC[y][x] += fragA[y] * fragB[x];
            //     }
            // }
            #pragma unroll
            for (int y = 0; y < REGM; y++) {
                for (int x = 0; x < REGN / 2; x++)
                {
                    reinterpret_cast<__half2*>(fragC[y])[x] = __hfma2(__half2half2(fragA[y]) ,reinterpret_cast<__half2*>(fragB)[x], reinterpret_cast<__half2*>(fragC[y])[x]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int y = 0; y < REGM; y++) 
    {
        #if (REGN % 8 == 0 && N % REGN == 0)
        #pragma unroll
        for (int x = 0; x < REGN / 8; x++)
            reinterpret_cast<float4 *>(&c[(blockIdx.x * TileM + threadIdx.x * REGM + y) * n + blockIdx.y * TileN + threadIdx.y * REGN])[x] = reinterpret_cast<float4 *>(&fragC[y][0])[x];
        #elif (REGN % 4 == 0 && N % REGN == 0)
        #pragma unroll
        for (int x = 0; x < REGN / 4; x++)
            reinterpret_cast<float2 *>(&c[(blockIdx.x * TileM + threadIdx.x * REGM + y) * n + blockIdx.y * TileN + threadIdx.y * REGN])[x] = reinterpret_cast<float2 *>(&fragC[y][0])[x];
        #else
        #pragma unroll
        for(int x = 0; x < REGN && blockIdx.x * TileM + threadIdx.x * REGM + y < M && blockIdx.y * TileN + threadIdx.y * REGN + x < N; x++)
        {
            c[(blockIdx.x * TileM + threadIdx.x * REGM + y) * n + blockIdx.y * TileN + threadIdx.y * REGN + x] = fragC[y][x];
        }
        #endif
    }
}


template <class T>
void cuBLASgemm(int m, int n, int k, const T *A, const T *B, T *C, Timeit &t)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    // malloc on device
    T *devPtrA, *devPtrB, *devPtrC;
    cudaMalloc((void **)&devPtrA, sizeof(T) * m * k);
    cudaMalloc((void **)&devPtrB, sizeof(T) * k * n);
    cudaMalloc((void **)&devPtrC, sizeof(T) * m * n);
    // copy A and B to device
    cudaMemcpy(devPtrA, A, sizeof(T) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrB, B, sizeof(T) * k * n, cudaMemcpyHostToDevice);

    T alpha = static_cast<T>(1);
    T beta = static_cast<T>(0);

    using GemmFunc = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, int, const void *, int, const void *, void *, int);
    GemmFunc gemm;
    if constexpr (std::is_same<T, float>::value)
    {
        gemm = reinterpret_cast<GemmFunc>(cublasSgemm);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        gemm = reinterpret_cast<GemmFunc>(cublasHgemm);
    }
    t.start();
    cublasStatus_t status = gemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, m, k,
                                 &alpha,
                                 devPtrB, n,
                                 devPtrA, k,
                                 &beta,
                                 devPtrC, n);
    t.stop();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("blas err = %d, str = %s\n", status, cublasGetStatusString(status));
    }
    // copy devPtrC to host
    cudaMemcpy(C, devPtrC, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
    // release memory on device
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
}


int main() {
    srand(1111);
    half *h_a, *h_b, *h_c;
    half *d_a, *d_b, *d_c;
    half *ground_truth;
    h_a = (half*)malloc( M * K * sizeof(half));
    h_b = (half*)malloc( K * N * sizeof(half));
    h_c = (half*)malloc( M * N * sizeof(half));
    ground_truth = (half*)malloc( M * N * sizeof(half));

    cudaMalloc((void**)&d_a, M * K * sizeof(half));
    cudaMalloc((void**)&d_b, K * N * sizeof(half));
    cudaMalloc((void**)&d_c, M * N * sizeof(half));

    gen_rand_data<half>(h_a, M * K);
    gen_rand_data<half>(h_b, K * N);

    cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice);
    Timeit t = Timeit();
    print_centered("cublas gemm", 100, '=');
    cuBLASgemm<half>(M, N, K, h_a, h_b, ground_truth, t);
    float cublas_FLOPS = t.get_FLOPS();
    printf("time: %f ms\n", t.elapsed_time);
    printf("FLOPS: %f FLOPS\n", t.get_FLOPS());
    

    print_centered("gmem_kernel", 100, '=');
    try{
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM, TileN);
        t.start();
        gmem_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "gmem_kernel");
        printf("time: %f ms\n", t.elapsed_time);
        printf("FLOPS: %f FLOPS(%.2f%%)\n", t.get_FLOPS(), t.get_FLOPS() / cublas_FLOPS * 100);
    }catch (const std::exception& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
    print_centered("tile_smem_kernel", 100, '=');
    try{
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM, TileN);
        t.start();
        tile_smem_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "tile_smem_kernel");
        printf("time: %f ms\n", t.elapsed_time);
        printf("FLOPS: %f FLOPS(%.2f%%)\n", t.get_FLOPS(), t.get_FLOPS() / cublas_FLOPS * 100);
    }catch (const std::exception& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
    print_centered("tile_smem_float4", 100, '=');
    try{
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM, TileN);
        t.start();
        tile_smem_float4_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "tile_smem_float4");
        printf("time: %f ms\n", t.elapsed_time);
        printf("FLOPS: %f FLOPS(%.2f%%)\n", t.get_FLOPS(), t.get_FLOPS() / cublas_FLOPS * 100);
    }catch (const std::exception& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
    print_centered("tile_smem_float4_tile_reg", 100, '=');
    try{
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        assert(TileM % REGM == 0);
        assert(TileN % REGN == 0);
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM/REGM, TileN/REGN);
        t.start();
        tile_smem_float4_tile_reg_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "tile_smem_float4_tile_reg");
        printf("time: %f ms\n", t.elapsed_time);
        printf("FLOPS: %f FLOPS(%.2f%%)\n", t.get_FLOPS(), t.get_FLOPS() / cublas_FLOPS * 100);
    }catch (const std::exception& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
    print_centered("tile_smem_float4_tile_reg_BT", 100, '=');
    try{
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        assert(TileM % REGM == 0);
        assert(TileN % REGN == 0);
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM/REGM, TileN/REGN);
        t.start();
        tile_smem_float4_tile_reg_BT_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "tile_smem_float4_tile_reg_BT");
        printf("time: %f ms\n", t.elapsed_time);
        printf("FLOPS: %f FLOPS(%.2f%%)\n", t.get_FLOPS(), t.get_FLOPS() / cublas_FLOPS * 100);
    }catch (const std::exception& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
    // print_matrix(h_a, M, K, "A");
    // print_matrix(h_b, K, N, "B");
    // print_matrix(h_c, M, N, "C");
    // print_matrix(ground_truth, M, N, "ground_truth");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

}