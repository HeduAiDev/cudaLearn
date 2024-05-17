
## Minimum requirements（这是cutlass库的需求）:
- Architecture: Volta
- Compiler: Must support at least C++17
- CUDA Toolkit version: 11.4

## 结构

~~~
.
├── 3rd
├── build
├── dist
├── CMakeLists.txt
├── LICENSE
├── copy.cu
├── filter.cu
├── gelu.cu
├── parallel_reduce.cu
├── queryProps.cu
└── readme.md
~~~

## 运行

~~~
cmake -S . -B build
cmake --build build
~~~

你还可以使用vscode的tasks来管理项目
- NVCC Run
    单文件（当前正在编辑的文件）编译并运行
- Nsight Compute
    使用Nsight Compute cli（ncu）调试当前文件并在Nsight Compute UI打开报告
- CMake Build ALL
    使用cmake编译整个项目
- Run Build File
    执行当前正在编辑的文件
