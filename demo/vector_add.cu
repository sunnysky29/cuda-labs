/**
 * CUDA 向量加法示例
 * 
 * 本程序在 GPU 上执行两个向量 A 和 B 的逐元素加法，结果存储在向量 C 中。
 *
 * 编译与运行:
 * 1. 将代码保存为 vector_add.cu
 * 2. 使用 nvcc 编译: 
 nvcc -o vector_add vector_add.cu
 * 3. 运行可执行文件: ./vector_add
for i in {1..1000}; do ./vector_add; done

 */

#include <iostream>
#include <cuda_runtime.h>

// CUDA 内核函数：在 GPU 上执行向量加法
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000; // 向量长度
    size_t size = N * sizeof(float);

    // 分配主机内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 设置线程和块的数量
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动 CUDA 内核
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 将结果从设备复制到主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Test PASSED\n";

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
