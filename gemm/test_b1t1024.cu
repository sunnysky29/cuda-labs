/*

nvcc  test_b1t1024.cu -o test_b1t1024  &&  ./test_b1t1024

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EPSILON 1e-5 // 校验精度


__global__ void MatrixMulKernel(float* Ad, float* Bd, float* Cd, int width)
{
    int offset = threadIdx.x;
    int row = offset / width;
    // 这个操作是用来快速计算 offset 除以 width 的余数（即取模运算）
    // ，但有一个重要前提：width 必须是 2 的幂次方（如 32, 64, 128 等）。
    int col  = offset & (width -1);

    float sum = 0;
    for (int i=0; i<width;i++){
        sum += Ad[row*width +i] * Bd[i*width + col];
    }
    Cd[row*width + col] = sum;

}

void MatrixMulOnHost(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;  //  0 --- N*N-1赋值
        }
    }
}

// 验证两个矩阵是否相等
int checkRes(float *hostRef, float *gpuRef, const int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > EPSILON) {
            printf("Mismatch at element %d: hostRef=%.6f, gpuRef=%.6f\n", 
                  i, hostRef[i], gpuRef[i]);
            return 1;
        }
        else {
            if (i<5) {  // 打印前5个结果元素
                printf("C[%d]=%.6f\n", 
                    i, gpuRef[i]);
            }

        }
    }
    printf("res match !! \n");
    return 0;
}

int main(void){
    srand(time(NULL));  // <-- 初始化随机数种子

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("using Device %d: %s\n", dev, deviceProp.name);

    cudaSetDevice(dev);

    int width = 1<<5;
    int size = width*width* sizeof(float);
    float *A, *B, *C, *gpuRef01,  *gpuRef02;

    A= (float*)malloc(size);
    B= (float*)malloc(size);
    C= (float*)malloc(size);

    gpuRef01= (float*)malloc(size);
    gpuRef02= (float*)malloc(size);

/*------------------初始化 A B------------------------*/
    clock_t iStart = clock();
    for (int i = 0; i<width; i++){
        for (int j = 0; j<width; j++){
            // 生成的是1.0到10.0之间的随机浮点数（整数部分）
            A[i*width+j] = float (rand() % 10 +1);
            B[i*width+j] = float (rand() % 10 +1);

        }
    }
    printf("Matrix A first 5 elements:\n");
    for (int i = 0; i < 5; i++) {
        printf("A[%d] = %.2f\n", i, A[i]);
    }
    double iElaps = (double)(clock() - iStart) / CLOCKS_PER_SEC;;
    printf("初始化：\t %.2f μs\n", iElaps*1e6);

/*------------------host 端矩阵乘------------------------*/
    iStart = clock();
    MatrixMulOnHost(A,B,C,width);
    iElaps = (double)(clock() - iStart) / CLOCKS_PER_SEC;;
    printf("host矩阵乘：\t %.2f μs\n", iElaps*1e6);


/*------------------开辟设备端内存空间------------------------*/
    float *Ad, *Bd, *Cd;
    cudaMalloc((void**)&Ad, size);
    cudaMalloc((void**)&Bd, size);
    cudaMalloc((void**)&Cd, size);
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

/*------------------核函数启动------------------------*/

    iStart = clock();
    MatrixMulKernel<<<1, 1024>>>(Ad,Bd,Cd,width);
    cudaDeviceSynchronize();
    iElaps = (double)(clock() - iStart) / CLOCKS_PER_SEC;;
    printf("device矩阵乘 on device <<<1, 1024>>>：\t %.2f μs\n", iElaps*1e6);


    cudaMemcpy(gpuRef01, Cd, size, cudaMemcpyDeviceToHost);
    // 调用处明确检查返回值：
    if (checkRes(C, gpuRef01, width) != 0) {
        printf("Error: GPU result does not match CPU reference!\n");
        // 可以在这里处理错误（如退出程序）
    }

    // 释放主机内存
    free(A);
    free(B);
    free(C);
    free(gpuRef01);
    free(gpuRef02);
    // 释放设备内存
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    return 0;
}