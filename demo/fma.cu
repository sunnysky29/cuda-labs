//  展示 cuda 的 FMA 
// FMA = Fused Multiply-Add

#include <stdio.h>

// Kernel: 每个线程执行一个 FMA 操作
__global__ void fmaKernel(float* a, float* b, float* c, float* d, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        d[i] = fmaf(a[i], b[i], c[i]);  // 使用 FMA 指令
    }
}

int main() {
    int n = 5;
    size_t size = n * sizeof(float);

    // 主机内存分配
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float c[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float d[5];

    float *d_a, *d_b, *d_c, *d_d;

    // 设备内存分配
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_d, size);

    // 主机 -> 设备拷贝
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

    // 启动 kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    printf(" <<< blockSize: %d ; numBlocks  : %d >>> \n", blockSize, numBlocks);
    fmaKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, d_d, n);

    // 设备 -> 主机拷贝
    cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);

    // 输出结果
    for (int i = 0; i < n; ++i) {
        printf("d[%d] = %f\n", i, d[i]);
    }

    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    return 0;
}