#include<iostream>
#include<cuda_runtime.h>
using namespace std;


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout << "可用GPU数量:" << deviceCount <<endl;
    if (deviceCount <=0) {
        cout << "没有可用的GPU" << endl;
        exit(0);
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cout << "GPU名称:" << deviceProp.name << endl;
    cout << "GPU计算能力:" << deviceProp.major << "." << deviceProp.minor << endl;
    cout << "GPU全局内存:" << deviceProp.totalGlobalMem/(1<<20) << "MB" << endl;
    cout << "GPU常量内存:" << deviceProp.totalConstMem/(1<<10) << "KB" << endl;
    cout << "SM共享内存大小:" << deviceProp.sharedMemPerMultiprocessor/(1<<10) << "KB" << endl;
    return 0;
}