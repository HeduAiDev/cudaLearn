//#ncu-add
#include <iostream>
#include <cute/tensor.hpp>
#include <cublas_v2.h>

using namespace cute;

#define M 2048
#define N 2048
#define K 1024

// half | float
using dtype = float;


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

template<typename T,int MatrixSize, int TileSize >
struct ArrT {
    using TYPE = std::conditional_t<
        (TileSize % (sizeof(float4) / sizeof(T)) == 0 && MatrixSize % TileSize == 0),
        float4,
        std::conditional_t<
            (TileSize % (sizeof(float2) / sizeof(T)) == 0 && MatrixSize % TileSize == 0),
            float2,
            std::conditional_t<
                (TileSize % (sizeof(float) / sizeof(T)) == 0 && MatrixSize % TileSize == 0),
                float,
                T>>>;
    static constexpr int ELEMENTS = []
    {
        if constexpr (TileSize % (sizeof(float4)/sizeof(T)) == 0 && MatrixSize % TileSize == 0)
            return (sizeof(float4)/sizeof(T));
        else if constexpr (TileSize % (sizeof(float2)/sizeof(T)) == 0 && MatrixSize % TileSize == 0)
            return (sizeof(float2)/sizeof(T));
        else if constexpr (TileSize % (sizeof(float)/sizeof(T)) == 0 && MatrixSize % TileSize == 0)
            return (sizeof(float)/sizeof(T));
        else
            return 1;
    }();
};


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
       return ((float)M * N * K * 2) / (elapsed_time) / 1e3;
    }
};

template<typename T1, typename T2>
__global__ void check_kernel(T1 *a, T2 *b, bool* flg, unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if(abs((float) (a[i] - b[i]) > 1e-6)) {
            printf("a[%d] = %.10f, b[%d] = %.10f\n", i, __half2float(a[i]), i, __half2float(b[i]));
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

template<typename T = half, int _TileM, int _TileN, int _TileK, int _KTileM, int _KTileN, int _KTileK>
__global__ void gmem_kernel(T *a, T *b, T *c, int m, int n, int k)
{
    int total_threads = gridDim.x * blockDim.x;
    #pragma unroll
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < m; row += total_threads)
    {
        #pragma unroll
        for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < n; col += total_threads) 
        {
            T sum = 0;
            for (int i = 0; i <k; i++)
            {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

// TileM,TileN
template<typename T = half, int _TileM, int _TileN, int _TileK, int _KTileM, int _KTileN, int _KTileK>
__global__ void tile_smem_kernel(T *a, T *b, T *c, const int m, const int n, const int k)
{
    __shared__ T sA[_TileM][_TileK];
    __shared__ T sB[_TileK][_TileN];
    int row = blockIdx.x * _TileM + threadIdx.x;
    int col = blockIdx.y * _TileN + threadIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    T sum = 0;

    #pragma unroll
    for (int tki = 0; tki < _KTileK; tki++)
    {
        #pragma unroll
        for(int i = tid; i < _TileM * _TileK; i += total_threads)        
        {
            int y = i % _TileM;
            int x = i / _TileM;
            sA[y][x] = a[(blockIdx.x * _TileM + y) * k + tki * _TileK + x];
        }
        #pragma unroll
        for(int i = tid; i < _TileK * _TileN; i += total_threads)        
        {
            int y = i % _TileK;
            int x = i / _TileK;
            sB[y][x] = b[(tki * _TileK + y) * n + blockIdx.y * _TileN + x];
        }
        __syncthreads();

        // 支持任意形状的 _TileK
        #pragma unroll
        for (int i = 0; i < min(k - tki * _TileK, _TileK); i++)
        {
            sum += sA[threadIdx.x][i] * sB[i][threadIdx.y];
        }
        __syncthreads();
    }
    if (row < m && col < n)
    {
        c[row * n + col] = sum;
    }
}

// TileM, TileN
template<typename T = half, int _TileM, int _TileN, int _TileK, int _KTileM, int _KTileN, int _KTileK>
__global__ void tile_smem_float4_kernel(T *a, T *b, T *c, int m, int n, int k)
{
    __shared__ T sA[_TileM][_TileK];
    __shared__ T sB[_TileK][_TileN];
    int row = blockIdx.x * _TileM + threadIdx.x;
    int col = blockIdx.y * _TileN + threadIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    T sum = 0;

    // sA
    using ArrT_a = typename ArrT<T, K, _TileK>::TYPE;
    constexpr int elements_a = ArrT<T, K, _TileK>::ELEMENTS;
    // sB
    using ArrT_b = typename ArrT<T, N, _TileN>::TYPE;
    constexpr int elements_b = ArrT<T, N, _TileN>::ELEMENTS;
   
    #pragma unroll
    for (int tki = 0; tki < _KTileK; tki++)
    {
        #pragma unroll
        for(int i = tid; i < _TileM * _TileK / elements_a; i += total_threads)        
        {
            int y = i % _TileM;
            int x = i / _TileM;
            reinterpret_cast<ArrT_a*>(&sA[y][x * elements_a])[0] = reinterpret_cast<ArrT_a*>(&a[(blockIdx.x * _TileM + y) * k + tki * _TileK + x * elements_a])[0];
        }
        #pragma unroll
        for(int i = tid; i < _TileK * _TileN / elements_b; i += total_threads)        
        {
            int y = i % _TileK;
            int x = i / _TileK;
            reinterpret_cast<ArrT_b*>(&sB[y][x * elements_b])[0] = reinterpret_cast<ArrT_b*>(&b[(tki * _TileK + y) * n + blockIdx.y * _TileN + x * elements_b])[0];
        }
        __syncthreads();

        // 允许k 不能被 TileK整除
        #pragma unroll
        for (int i = 0; i < min(k - tki * _TileK, _TileK); i++)
        {
            sum += sA[threadIdx.x][i] * sB[i][threadIdx.y];
        }
        __syncthreads();
    }
    if (row < m && col < n)
    {
        c[row * n + col] = sum;
    }
}


// TileM/REGM, TileN/REGN
template<typename T = half, int _TileM, int _TileN, int _TileK, int _KTileM, int _KTileN, int _KTileK, int _REGM, int _REGN>
__global__ void tile_smem_float4_tile_reg_kernel(T * __restrict__ a, T * __restrict__ b, T * __restrict__ c, int m, int n, int k)
{
    __shared__ T sA[_TileM][_TileK];
    __shared__ T sB[_TileK][_TileN];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    T fragA[_REGM];
    T fragB[_REGN];
    T fragC[_REGM][_REGN] = {0};
    if (threadIdx.x >= _TileM/_REGM || threadIdx.y >= _TileN/_REGN) return;

    // sA
    using ArrT_a = typename ArrT<T, K, _TileK>::TYPE;
    constexpr int elements_a = ArrT<T, K, _TileK>::ELEMENTS;
    // sB
    using ArrT_b = typename ArrT<T, N, _TileN>::TYPE;
    constexpr int elements_b = ArrT<T, N, _TileN>::ELEMENTS;
    // REG
    using ArrT_c = typename ArrT<T, _REGN, _REGN>::TYPE;
    constexpr int elements_c = ArrT<T, _REGN, _REGN>::ELEMENTS;

    #pragma unroll
    for (int tki = 0; tki < _KTileK; tki++)
    {
        #pragma unroll
        for(int i = tid; i < _TileM * _TileK / elements_a; i += total_threads)        
        {
            int y = i % _TileM;
            int x = i / _TileM;
            reinterpret_cast<ArrT_a*>(&sA[y][x * elements_a])[0] = reinterpret_cast<ArrT_a*>(&a[(blockIdx.x * _TileM + y) * k + tki * _TileK + x * elements_a])[0];
        }
        #pragma unroll
        for(int i = tid; i < _TileK * _TileN / elements_b; i += total_threads)        
        {
            int y = i % _TileK;
            int x = i / _TileK;
            reinterpret_cast<ArrT_b*>(&sB[y][x * elements_b])[0] = reinterpret_cast<ArrT_b*>(&b[(tki * _TileK + y) * n + blockIdx.y * _TileN + x * elements_b])[0];
        }
        __syncthreads();

        // 允许k 不能被 _TileK整除
        #pragma unroll
        for (int i = 0; i < min(k - tki * _TileK, _TileK); i++)
        {
            #pragma unroll
            for(int j = 0; j < _REGM; j++)    
            {
                fragA[j] = sA[threadIdx.x * _REGM + j][i];
            } 
            #pragma unroll
            for (int j = 0; j < _REGN / elements_c; j++)
                reinterpret_cast<ArrT_c *>(fragB)[j] = reinterpret_cast<ArrT_c *>(&sB[i][threadIdx.y * _REGN])[j];
            if constexpr (!std::is_same<T, __half>::value)
            {
                #pragma unroll
                for (int y = 0; y < _REGM; y++)
                {
                    for (int x = 0; x < _REGN; x++)
                    {
                        fragC[y][x] += fragA[y] * fragB[x];
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int y = 0; y < _REGM; y++)
                {
                    for (int x = 0; x < _REGN / 2; x++)
                    {
                        reinterpret_cast<__half2 *>(fragC[y])[x] = __hfma2(__half2half2(fragA[y]), reinterpret_cast<__half2 *>(fragB)[x], reinterpret_cast<__half2 *>(fragC[y])[x]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int y = 0; y < _REGM; y++) 
    {
        if constexpr (_REGN % (sizeof(float4)/sizeof(T)) == 0 && N % _REGN == 0)
        {
            #pragma unroll
            for (int x = 0; x < _REGN / (sizeof(float4)/sizeof(T)); x++)
                reinterpret_cast<float4 *>(&c[(blockIdx.x * _TileM + threadIdx.x * _REGM + y) * n + blockIdx.y * _TileN + threadIdx.y * _REGN])[x] = reinterpret_cast<float4 *>(&fragC[y][0])[x];
        }
        else if constexpr (_REGN % (sizeof(float2)/sizeof(T)) == 0 && N % _REGN == 0)
        {
            #pragma unroll
            for (int x = 0; x < _REGN / (sizeof(float2)/sizeof(T)); x++)
                reinterpret_cast<float2 *>(&c[(blockIdx.x * _TileM + threadIdx.x * _REGM + y) * n + blockIdx.y * _TileN + threadIdx.y * _REGN])[x] = reinterpret_cast<float2 *>(&fragC[y][0])[x];
        }
        else
        {
            #pragma unroll
            for (int x = 0; x < _REGN && blockIdx.x * _TileM + threadIdx.x * _REGM + y < M && blockIdx.y * _TileN + threadIdx.y * _REGN + x < N; x++)
            {
                c[(blockIdx.x * _TileM + threadIdx.x * _REGM + y) * n + blockIdx.y * _TileN + threadIdx.y * _REGN + x] = fragC[y][x];
            }
        }
    }
}

// TileM/REGM, TileN/REGN
template<typename T = half, int _TileM, int _TileN, int _TileK, int _KTileM, int _KTileN, int _KTileK, int _REGM, int _REGN>
__global__ void tile_smem_float4_tile_reg_BT_kernel(T * __restrict__ a, T * __restrict__ b, T * __restrict__ c, int m, int n, int k)
{
    __shared__ T sA[_TileK][_TileM];
    __shared__ T sB[_TileK][_TileN];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    T fragA[_REGM];
    T fragB[_REGN];
    T fragC[_REGM][_REGN] = {0};
    if (threadIdx.x >= _TileM/_REGM || threadIdx.y >= _TileN/_REGN) return;
    // sA
    using ArrT_a = typename ArrT<T, K, _TileK>::TYPE;
    constexpr int elements_a = ArrT<T, K, _TileK>::ELEMENTS;
    // sB
    using ArrT_b = typename ArrT<T, N, _TileN>::TYPE;
    constexpr int elements_b = ArrT<T, N, _TileN>::ELEMENTS;
    // _REGN
    using ArrT_cn = typename ArrT<T, _REGN, _REGN>::TYPE;
    constexpr int elements_cn = ArrT<T, _REGN, _REGN>::ELEMENTS;
    // _REGM
    using ArrT_cm = typename ArrT<T, _REGM, _REGM>::TYPE;
    constexpr int elements_cm = ArrT<T, _REGM, _REGM>::ELEMENTS;

    #pragma unroll
    for (int tki = 0; tki < _KTileK; tki++)
    {
        #pragma unroll
        for(int i = tid; i < _TileM * _TileK / elements_a; i += total_threads)        
        {
            T tmp_a[elements_a];
            int y = i % _TileM;
            int x = i / _TileM;
            reinterpret_cast<ArrT_a*>(tmp_a)[0] = reinterpret_cast<ArrT_a*>(&a[(blockIdx.x * _TileM + y) * k + tki * _TileK + x * elements_a])[0];
            #pragma unroll
            for(int j = 0; j < elements_a; j++)    
            {
                sA[x * elements_a + j][y] = tmp_a[j];
            }
        }
        #pragma unroll
        for(int i = tid; i < _TileK * _TileN / elements_b; i += total_threads)        
        {
            int y = i % _TileK;
            int x = i / _TileK;
            reinterpret_cast<ArrT_b*>(&sB[y][x * elements_b])[0] = reinterpret_cast<ArrT_b*>(&b[(tki * _TileK + y) * n + blockIdx.y * _TileN + x * elements_b])[0];
            
        }
        __syncthreads();

        // 允许k 不能被 TileK整除
        #pragma unroll
        for (int i = 0; i < min(k - tki * _TileK, _TileK); i++)
        {
            #pragma unroll
            for (int j = 0; j < _REGM / elements_cm; j++)
                reinterpret_cast<ArrT_cm *>(fragA)[j] = reinterpret_cast<ArrT_cm *>(&sA[i][threadIdx.x * _REGM])[j];
            #pragma unroll
            for (int j = 0; j < _REGN / elements_cn; j++)
                reinterpret_cast<ArrT_cn *>(fragB)[j] = reinterpret_cast<ArrT_cn *>(&sB[i][threadIdx.y * _REGN])[j];
            if constexpr (!std::is_same<T, __half>::value)
            {
                #pragma unroll
                for (int y = 0; y < _REGM; y++)
                {
                    for (int x = 0; x < _REGN; x++)
                    {
                        fragC[y][x] += fragA[y] * fragB[x];
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int y = 0; y < _REGM; y++)
                {
                    for (int x = 0; x < _REGN / 2; x++)
                    {
                        reinterpret_cast<__half2 *>(fragC[y])[x] = __hfma2(__half2half2(fragA[y]), reinterpret_cast<__half2 *>(fragB)[x], reinterpret_cast<__half2 *>(fragC[y])[x]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int y = 0; y < _REGM; y++) 
    {
        if constexpr (_REGN % (sizeof(float4)/sizeof(T)) == 0 && N % _REGN == 0)
        {
            #pragma unroll
            for (int x = 0; x < _REGN / (sizeof(float4)/sizeof(T)); x++)
                reinterpret_cast<float4 *>(&c[(blockIdx.x * _TileM + threadIdx.x * _REGM + y) * n + blockIdx.y * _TileN + threadIdx.y * _REGN])[x] = reinterpret_cast<float4 *>(&fragC[y][0])[x];
        }
        else if constexpr (_REGN % (sizeof(float2)/sizeof(T)) == 0 && N % _REGN == 0)
        {
            #pragma unroll
            for (int x = 0; x < _REGN / (sizeof(float2)/sizeof(T)); x++)
                reinterpret_cast<float2 *>(&c[(blockIdx.x * _TileM + threadIdx.x * _REGM + y) * n + blockIdx.y * _TileN + threadIdx.y * _REGN])[x] = reinterpret_cast<float2 *>(&fragC[y][0])[x];
        }
        else
        {
            #pragma unroll
            for (int x = 0; x < _REGN && blockIdx.x * _TileM + threadIdx.x * _REGM + y < M && blockIdx.y * _TileN + threadIdx.y * _REGN + x < N; x++)
            {
                c[(blockIdx.x * _TileM + threadIdx.x * _REGM + y) * n + blockIdx.y * _TileN + threadIdx.y * _REGN + x] = fragC[y][x];
            }
        }
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
    dtype *h_a, *h_b, *h_c;
    dtype *d_a, *d_b, *d_c;
    dtype *ground_truth;
    h_a = (dtype*)malloc( M * K * sizeof(dtype));
    h_b = (dtype*)malloc( K * N * sizeof(dtype));
    h_c = (dtype*)malloc( M * N * sizeof(dtype));
    ground_truth = (dtype*)malloc( M * N * sizeof(dtype));

    cudaMalloc((void**)&d_a, M * K * sizeof(dtype));
    cudaMalloc((void**)&d_b, K * N * sizeof(dtype));
    cudaMalloc((void**)&d_c, M * N * sizeof(dtype));

    gen_rand_data<dtype>(h_a, M * K);
    gen_rand_data<dtype>(h_b, K * N);

    cudaMemcpy(d_a, h_a, M * K * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(dtype), cudaMemcpyHostToDevice);
    Timeit t = Timeit();
    // warmup
    cuBLASgemm<dtype>(M, N, K, h_a, h_b, ground_truth, t);
    // warmup end
    print_centered("cublas gemm", 100, '=');
    cuBLASgemm<dtype>(M, N, K, h_a, h_b, ground_truth, t);
    float cublas_FLOPS = t.get_FLOPS();
    printf("time: %f ms\n", t.elapsed_time);
    printf("throughput: %f KFLOPS\n", t.get_FLOPS() / 1e3);
    

    print_centered("gmem_kernel", 100, '=');
    try{
        constexpr int TileM = 8;
        constexpr int TileN = 16;
        constexpr int TileK = 1; //ignore

        constexpr int KTileM = (((M) + (TileM - 1))/TileM);
        constexpr int KTileK = (((K) + (TileK - 1))/TileK);
        constexpr int KTileN = (((N) + (TileN - 1))/TileN);

        // constexpr int REGM = 8;
        // constexpr int REGN = 8;
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM, TileN);
        t.start();
        gmem_kernel<dtype, TileM, TileN, TileK, KTileM, KTileN, KTileK><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(dtype), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "gmem_kernel");
        printf("config: grid(%d x %d), block(%d x %d)\n", grid.x, grid.y, block.x, block.y);
        printf("time: %f ms\n", t.elapsed_time);
        printf("throughput: %f KFLOPS(%.2f%%)\n", t.get_FLOPS() / 1e3, t.get_FLOPS() / cublas_FLOPS * 100);
    }catch (const std::exception& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
    print_centered("tile_smem_kernel", 100, '=');
    try{
        constexpr int TileM = 8;
        constexpr int TileN = 16;
        constexpr int TileK = 4;

        constexpr int KTileM = (((M) + (TileM - 1))/TileM);
        constexpr int KTileK = (((K) + (TileK - 1))/TileK);
        constexpr int KTileN = (((N) + (TileN - 1))/TileN);

        // constexpr int REGM = 8;
        // constexpr int REGN = 8;
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM, TileN);
        t.start();
        tile_smem_kernel<dtype, TileM, TileN, TileK, KTileM, KTileN, KTileK><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(dtype), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "tile_smem_kernel");
        printf("config: grid(%d x %d), block(%d x %d)\n", grid.x, grid.y, block.x, block.y);
        printf("time: %f ms\n", t.elapsed_time);
        printf("throughput: %f KFLOPS(%.2f%%)\n", t.get_FLOPS() / 1e3, t.get_FLOPS() / cublas_FLOPS * 100);
    }catch (const std::exception& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
    print_centered("tile_smem_float4", 100, '=');
    try{
        constexpr int TileM = 16;
        constexpr int TileN = 8;
        constexpr int TileK = 8;

        constexpr int KTileM = (((M) + (TileM - 1))/TileM);
        constexpr int KTileK = (((K) + (TileK - 1))/TileK);
        constexpr int KTileN = (((N) + (TileN - 1))/TileN);

        // constexpr int REGM = 8;
        // constexpr int REGN = 8;
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM, TileN);
        t.start();
        tile_smem_float4_kernel<dtype, TileM, TileN, TileK, KTileM, KTileN, KTileK><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(dtype), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "tile_smem_float4");
        printf("config: grid(%d x %d), block(%d x %d)\n", grid.x, grid.y, block.x, block.y);
        printf("time: %f ms\n", t.elapsed_time);
        printf("throughput: %f KFLOPS(%.2f%%)\n", t.get_FLOPS() / 1e3, t.get_FLOPS() / cublas_FLOPS * 100);
    }catch (const std::exception& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
    print_centered("tile_smem_float4_tile_reg", 100, '=');
    try{
        constexpr int TileM = 256;
        constexpr int TileN = 128;
        constexpr int TileK = 16;

        constexpr int KTileM = (((M) + (TileM - 1))/TileM);
        constexpr int KTileK = (((K) + (TileK - 1))/TileK);
        constexpr int KTileN = (((N) + (TileN - 1))/TileN);

        constexpr int REGM = 16;
        constexpr int REGN = 16;
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        assert(TileM % REGM == 0);
        assert(TileN % REGN == 0);
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM/REGM, TileN/REGN);
        t.start();
        tile_smem_float4_tile_reg_kernel<dtype, TileM, TileN, TileK, KTileM, KTileN, KTileK, REGM, REGN><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(dtype), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "tile_smem_float4_tile_reg");
        printf("config: grid(%d x %d), block(%d x %d)\n", grid.x, grid.y, block.x, block.y);
        printf("time: %f ms\n", t.elapsed_time);
        printf("throughput: %f KFLOPS(%.2f%%)\n", t.get_FLOPS() / 1e3, t.get_FLOPS() / cublas_FLOPS * 100);
    }catch (const std::exception& e) {
        std::cerr << "Caught an exception: " << e.what() << std::endl;
    }
    print_centered("tile_smem_float4_tile_reg_BT", 100, '=');
    try{
        constexpr int TileM = 256;
        constexpr int TileN = 128;
        constexpr int TileK = 8;

        constexpr int KTileM = (((M) + (TileM - 1))/TileM);
        constexpr int KTileK = (((K) + (TileK - 1))/TileK);
        constexpr int KTileN = (((N) + (TileN - 1))/TileN);

        constexpr int REGM = 16;
        constexpr int REGN = 16;
        // 支持任意形状 M,N,K 及 TileM,TileN,TileK
        assert(TileM % REGM == 0);
        assert(TileN % REGN == 0);
        dim3 grid(KTileM, KTileN);
        dim3 block(TileM/REGM, TileN/REGN);
        t.start();
        tile_smem_float4_tile_reg_BT_kernel<dtype, TileM, TileN, TileK, KTileM, KTileN, KTileK, REGM, REGN><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        t.stop();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_c, d_c, M * N * sizeof(dtype), cudaMemcpyDeviceToHost));
        check<(M * N + 127) / 128, 128>(h_c, ground_truth, M * N, "tile_smem_float4_tile_reg_BT");
        printf("config: grid(%d x %d), block(%d x %d)\n", grid.x, grid.y, block.x, block.y);
        printf("time: %f ms\n", t.elapsed_time);
        printf("throughput: %f KFLOPS(%.2f%%)\n", t.get_FLOPS() / 1e3, t.get_FLOPS() / cublas_FLOPS * 100);
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