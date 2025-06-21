/*
CPU 矩阵乘法测试（原生实现 vs OpenBLAS）
编译命令：
    gcc -O3 ./gemm_host.c -lopenblas -o gemm_test && ./gemm_test

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // 添加memset需要的头文件
#include <time.h>
#include <math.h>
#include <cblas.h>   // OpenBLAS 头文件

#define N 32        // 方阵大小
#define EPSILON 1e-5 // 校验精度

// 原生矩阵乘法, 注意这里是
void matrix_multiply(int n, float *A, float *B, float *C) {
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

// 生成随机矩阵
void init_random_matrix(int n, float *matrix) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// 验证两个矩阵是否相等
int verify_results(int n, float *C1, float *C2) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(C1[i] - C2[i]) > EPSILON) {
            printf("Mismatch at element %d: C1=%.6f, C2=%.6f\n", 
                  i, C1[i], C2[i]);
            return 0;
        }
        else {
            if (i<5) {  // 打印前5个结果元素
                printf("C[%d]=%.6f\n", 
                    i, C1[i]);
            }

        }
    }
    return 1;
}

int main() {
    srand(time(NULL));  // <-- 初始化随机数种子
    
    // 分配内存
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C_native = (float *)malloc(N * N * sizeof(float));
    float *C_blas = (float *)malloc(N * N * sizeof(float));
    
    // 初始化随机矩阵
    init_random_matrix(N, A);
    init_random_matrix(N, B);
    
    // 清零结果矩阵
    memset(C_native, 0, N * N * sizeof(float));
    memset(C_blas, 0, N * N * sizeof(float));

    // 测试原生实现
    clock_t start = clock();
    matrix_multiply(N, A, B, C_native);
    double native_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    // 测试OpenBLAS实现
    start = clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C_blas, N);
    double blas_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    // 打印性能结果
    printf("Performance Results (Matrix size: %dx%d):\n", N, N);
    printf(" - Native implementation: %.6f seconds\n", native_time);
    printf(" - OpenBLAS implementation: %.6f seconds\n", blas_time);
    printf(" - Speedup: %.2fx\n", native_time / blas_time);
    
    // 验证结果一致性
    printf("\nVerifying results...\n");
    if (verify_results(N, C_native, C_blas)) {
        printf("Results match!\n");
    } else {
        printf("Results differ!\n");
    }
    
    // 释放内存
    free(A);
    free(B);
    free(C_native);
    free(C_blas);
    
    return 0;
}