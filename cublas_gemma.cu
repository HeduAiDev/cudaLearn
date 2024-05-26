#include <iostream>
#include <cublas_v2.h>

#define CUBLAS_CHECK(stat)                                                        \
    {                                                                             \
        cublasStatus_t err = stat;                                                \
        if (err != CUBLAS_STATUS_SUCCESS)                                         \
        {                                                                         \
            printf("blas err = %d, str = %s\n", err, cublasGetStatusString(err)); \
        }                                                                         \
    }

void init_matrix(float *a, int size) 
{
    for (int i = 0; i < size; i++)
    {
        a[i] = (int)(rand() % 10 - 5);
    }
}

void print_matrix(float *a, int rows, int cols, std::string prefix = "") 
{
    printf("Matrix %s:\n", prefix.c_str());
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%5.0f", a[i * cols + j]);
        }
        printf("\n");
    }
}

// convert row-major matrix to column-major matrix
void row2col(const float *a, int rows, int cols, float *b)
{
    int size = rows * cols;
    for (int i = 0; i < size; i++)
    {
        int row = i / cols;
        int col = i % cols;
        b[col * rows + row] = a[i];
    }
}

// convert column-major matrix to row-major matrix
void col2row(const float *a, int rows, int cols, float *b)
{
    int size = rows * cols;
    for (int i = 0; i < size; i++)
    {
        int col = i / rows;
        int row = i % rows;
        b[row * cols + col] = a[i];
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

void transpose(float *a, int rows, int cols, float *b)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            b[j * rows + i] = a[i * cols + j];
        }
    }
}



int main()
{
    srand(1111);
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    int m = 3;
    int n = 2;
    int k = 4;

    A = (float*)malloc(m * k * sizeof(float));
    B = (float*)malloc(k * n * sizeof(float));
    C = (float*)malloc(m * n * sizeof(float));

    init_matrix(A, m * k);
    init_matrix(B, k * n);
    float *B_nxk = (float *)malloc(n * k * sizeof(float));
    transpose(B, k, n, B_nxk);
    print_matrix(A, m, k, "A(mxk)");
    print_matrix(B, k, n, "B(kxn)");
    print_matrix(B_nxk, n, k, "B(nxk)");

    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;
    //1. OP_N(B(kxn))(nxk) x OP_N(A(mxk))(kxm) = C.T(nxm)
    print_centered("OP_N(B(kxn))(nxk) x OP_N(A(mxk))(kxm) = C.T(nxm)", 100, '=');
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
        cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(C, m, n, "col major C.T(nxm) = row major read C(mxn)");
        cublasDestroy(handle);
    }
    //2. OP_N(row2col(A)(mxk))(mxk) x OP_N(row2col(B)(kxn))(kxn) = C(mxn) -> col2row(C) = C.T(nxm)
    print_centered("row2col(A)(mxk) x row2col(B)(kxn) = C(mxn) -> col2row(C) = C.T(nxm)",100,'=');
    {
        float *A_col = (float *)malloc(m * k * sizeof(float));
        float *B_col = (float *)malloc(k * n * sizeof(float));
        float *C_col = (float *)malloc(m * n * sizeof(float));
        row2col(A, m, k, A_col);
        row2col(B, k, n, B_col);
        // print_matrix(A_col, m, k, "A_col(mxk)");
        // print_matrix(B_col, k, n, "B_col(kxn)");
        cudaMemcpy(d_A, A_col, m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B_col, k * n * sizeof(float), cudaMemcpyHostToDevice);
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
        cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        col2row(C, m, n, C_col);
        print_matrix(C, m, n, "raw C(mxn)");
        print_matrix(C_col, m, n, "col2row(C)(mxn)");
        cublasDestroy(handle);
    }
    // 3.OP_T(A(mxk))(mxk) x OP_T(B(kxn))(kxn) = C(mxn) -> col2row(C) = C.T(nxm)
    print_centered("OP_T(A(mxk)))(mxk) x OP_T(B(kxn))(kxn) = C(mxn) -> col2row(C) = C.T(nxm)", 100, '=');
    {
        float *C_col = (float *)malloc(m * n * sizeof(float));
        cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, m);
        cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        col2row(C, m, n, C_col);
        print_matrix(C, m, n, "raw C(mxn)");
        print_matrix(C_col, m, n, "col2row(C)(mxn)");
        cublasDestroy(handle);
    }
    // 4. OP_T(A(mxk))(mxk) x OP_N(B(nxk))(kxn) = C(mxn) -> col2row(C)
    print_centered("OP_T(A(mxk))(mxk) x OP_N(B(nxk))(kxn) = C(mxn) -> col2row(C) = C.T(nxm)", 100, '=');
    {
        
        float *C_col = (float *)malloc(m * n * sizeof(float));
        cudaMemcpy(d_B, B_nxk, n * k * sizeof(float), cudaMemcpyHostToDevice);
        cublasHandle_t handle;
        cublasCreate(&handle);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, d_A, k, d_B, k, &beta, d_C, m););
        cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        col2row(C, m, n, C_col);
        print_matrix(C, m, n, "raw C(mxn)");
        print_matrix(C_col, m, n, "col2row(C)(mxn)");
        cublasDestroy(handle);
    }
    // 5.OP_T(B(nxk))(nxk) x OP_N(A(mxk))(kxm) = C.T(nxm)
    print_centered("OP_T(B(nxk))(nxk) x OP_N(A(mxk))(kxm) = C.T(nxm)", 100, '=');
    {
        cudaMemcpy(d_B, B_nxk, n * k * sizeof(float), cudaMemcpyHostToDevice);
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, d_B, k, d_A, k, &beta, d_C, n);
        cudaMemcpy(C, d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(C, m, n, "col major C.T(nxm) = row major read C(mxn)");
        cublasDestroy(handle);
    }
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(A);
        free(B);
        free(C);
}