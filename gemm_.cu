// File: gemm_compare.cu
// CUDA Matrix Multiplication: Naive vs Tiled with Shared Memory
// Compile: nvcc -O3 -arch=sm_75 gemm_compare.cu -o gemm_compare
// Run: ./gemm_compare
//  矩阵乘法 share mem 优化

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>

// ========================
// 配置参数
// ========================
#define N 512           // 矩阵大小 (N x N)，可改为 1024 测试
#define TILE_M 16       // Shared memory tile size in M
#define TILE_N 16       // Shared memory tile size in N
#define TILE_K 8        // Shared memory tile size in K


// ========================
// CPU 版本矩阵乘法（黄金标准）
// C = A * B, 其中 A, B, C 都是 N x N 矩阵，行优先存储
// ========================
void matmul_cpu(const float *A, const float *B, float *C, int N_) {
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N_; ++k) {
                sum += A[i * N_ + k] * B[k * N_ + j];
            }
            C[i * N_ + j] = sum;
        }
    }
}

// ========================
// 朴素版本 Kernel
// ========================
__global__ void matmul_naive(const float *A, const float *B, float *C, int size) {
    // ------------------> x 
    // |
    // |
    // |
    // |
    // |
    // |
    // |
    //\|/
    // y
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // y方向 -> 行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x方向 -> 列索引

    if (row < size && col < size) {
        float c = 0.0f;
        for (int k = 0; k < size; ++k) {
            c += A[row * size + k] * B[k * size + col];  // 行优先访问,/ 全局内存读取
        }
        C[row * size + col] = c;
    }
}

// ========================
// 优化版本 Kernel: Tiled + Shared Memory
// 每个线程块处理 TILE_M x TILE_N，使用 shared memory
// ========================
__global__ void matmul_tiled1(const float *A, const float *B, float *C, int size) {
    // Shared memory tiles
    __shared__ float As[TILE_M][TILE_K];  // 缓存 A 的一块: [TILE_M 行] × [TILE_K 列]
    __shared__ float Bs[TILE_K][TILE_N];  // 缓存 B 的一块: [TILE_K 行] × [TILE_N 列]

    // === 1. 线程身份识别 ===
    int tx = threadIdx.x;        // 块内列索引 (0 ~ TILE_N-1)
    int ty = threadIdx.y;        // 块内行索引 (0 ~ TILE_M-1)

    int block_col = blockIdx.x * TILE_N;   // 当前 block 负责的起始列
    int block_row = blockIdx.y * TILE_M;   // 当前 block 负责的起始行

    int global_row = block_row + ty;       // 当前线程计算的全局行号
    int global_col = block_col + tx;       // 当前线程计算的全局列号

    float sum = 0.0f;  // 累加器，用于计算 C[global_row][global_col]

    // === 2. 分段处理 K 维度（k_start 是每次加载的起始 k）===
    for (int k_start = 0; k_start < size; k_start += TILE_K) {

        // k_global > size 发生在 矩阵大小 N 不能被 TILE_K 整除 的情况下，
        // 尤其是在最后一块（last tile） 加载时。 
        
        // --- 加载 A[global_row][k_start : k_start + TILE_K] 到 shared memory ---
        if (ty < TILE_M && tx < TILE_K && global_row < size) {
            int k_global = k_start + tx;  // 全局 k 索引
            As[ty][tx] = (k_global < size) ? A[global_row * size + k_global] : 0.0f;
        }

        // --- 加载 B[k_start : k_start + TILE_K][global_col] 到 shared memory ---
        if (tx < TILE_N && ty < TILE_K && global_col < size) {
            int k_global = k_start + ty;  // 全局 k 索引
            Bs[ty][tx] = (k_global < size) ? B[k_global * size + global_col] : 0.0f;
        }

        __syncthreads(); // === 同步：确保所有线程完成加载 ===

        // === 计算部分和：使用当前 tile 的 As 和 Bs ===
        #pragma unroll  // 告诉编译器：把这个循环“展开”（unroll），以减少循环开销、提高性能。 
        for (int k_local = 0; k_local < TILE_K; ++k_local) {
            sum += As[ty][k_local] * Bs[k_local][tx];
        }

        // === 同步：为下一轮加载做准备（防止数据竞争）===
        __syncthreads();
    }

    // === 3. 写回结果 ===
    if (global_row < size && global_col < size) {
        C[global_row * size + global_col] = sum;
    }
}



// ========================
// 初始化矩阵
// ========================
void init_matrix(float *h_A, int size) {
    for (int i = 0; i < size * size; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }

    printf("\n--- First 5x5 block of host matrix ---\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.3f ", h_A[i * N + j]);
        }
        printf("\n");
    }
}

// ========================
// CUDA 错误检查宏
// ========================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ========================
// 主函数
// ========================
int main() {
    float *h_A, *h_B, *h_C_naive, *h_C_tiled;
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;

    size_t bytes = N * N * sizeof(float);

    // 主机内存分配
    h_A         = (float*)malloc(bytes);
    h_B         = (float*)malloc(bytes);
    h_C_naive   = (float*)malloc(bytes);
    h_C_tiled   = (float*)malloc(bytes);

    // 初始化
    srand(123);
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    // 设备内存分配
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C_naive, bytes));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, bytes));

    // 拷贝数据到 GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 定义线程块和网格
    dim3 block(TILE_M, TILE_N);  // (x,y)
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    // 👉 打印维度信息
    printf("\n=== CUDA Execution Configuration ===\n");
    printf("Matrix size        : %d x %d\n", N, N);
    printf("Tile size (M, N, K): (%d, %d, %d)\n", TILE_M, TILE_N, TILE_K);
    printf("Thread block       : (%d, %d)\n", block.x, block.y);
    printf("Grid size          : (%d, %d)\n", grid.x, grid.y);
    printf("Total blocks       : %d\n", grid.x * grid.y);
    printf("Threads per block  : %d\n", block.x * block.y);
    printf("Total threads      : %d\n", grid.x * grid.y * block.x * block.y);
    printf("====================================\n\n");

    // 创建 CUDA event 用于精确计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ------------------- 测量朴素版本 -------------------
    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C_naive, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_naive;
    CUDA_CHECK(cudaEventElapsedTime(&time_naive, start, stop));

    // ------------------- 测量优化版本 -------------------
    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled1<<<grid, block>>>(d_A, d_B, d_C_tiled, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_tiled;
    CUDA_CHECK(cudaEventElapsedTime(&time_tiled, start, stop));

    // 拷贝结果回 CPU
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C_tiled, bytes, cudaMemcpyDeviceToHost));


    // 简单验证：检查所有元素是否接近
    bool pass = true;
    int first_error = -1;
    for (int i = 0; i < N * N; i++) {
        float diff = fabs(h_C_naive[i] - h_C_tiled[i]);
        if (diff > 1e-4) {
            pass = false;
            first_error = i;
            break;
        }
    }


    // ========== 打印矩阵调试信息 ==========
    if (!pass && first_error != -1) {
        int row = first_error / N;
        int col = first_error % N;
        printf("\n--- ERROR DETECTED ---\n");
        printf("First mismatch at C[%d][%d] (index %d):\n", row, col, first_error);
        printf("  Naive  = %.6f\n", h_C_naive[first_error]);
        printf("  Tiled  = %.6f\n", h_C_tiled[first_error]);
        printf("  Diff   = %.6f\n", fabs(h_C_naive[first_error] - h_C_tiled[first_error]));
    }

    // ========== 打印前 5x5 子矩阵对比 ==========
    printf("\n--- First 5x5 block of C (Naive) ---\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.3f ", h_C_naive[i * N + j]);
        }
        printf("\n");
    }

    printf("\n--- First 5x5 block of C (Tiled) ---\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.3f ", h_C_tiled[i * N + j]);
        }
        printf("\n");
    }

    // ------------------- CPU 计算参考结果 -------------------
    float *h_C_cpu = (float*)malloc(bytes);
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    printf("\n--- First 5x5 block of C (CPU Reference) ---\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.3f ", h_C_cpu[i * N + j]);
        }
        printf("\n");
    }

    // ========== 打印出错位置附近的上下文（3x3 区域）==========
    if (!pass && first_error != -1) {
        int row = first_error / N;
        int col = first_error % N;
        int start_i = (row > 1) ? row - 1 : 0;
        int start_j = (col > 1) ? col - 1 : 0;
        int end_i = (row < N-1) ? row + 2 : N;
        int end_j = (col < N-1) ? col + 2 : N;

        printf("\n--- Context around mismatch C[%d][%d] ---\n", row, col);
        printf("         ");
        for (int j = start_j; j < end_j; j++) printf("  Naive[%d]   ", j);
        printf("   |    ");
        for (int j = start_j; j < end_j; j++) printf("  Tiled[%d]   ", j);
        printf("\n");

        for (int i = start_i; i < end_i; i++) {
            printf("Row %2d: ", i);
            for (int j = start_j; j < end_j; j++) {
                printf("%9.4f ", h_C_naive[i * N + j]);
            }
            printf("   |   ");
            for (int j = start_j; j < end_j; j++) {
                printf("%9.4f ", h_C_tiled[i * N + j]);
            }
            printf("\n");
        }
    }
    
    // 输出性能
    printf("Matrix Size: %d x %d\n", N, N);
    printf("Naive Version Time: %.4f ms\n", time_naive);
    printf("Tiled  Version Time: %.4f ms\n", time_tiled);
    printf("Speedup: %.2fx\n", time_naive / time_tiled);
    printf("Correctness: %s\n", pass ? "PASS √" : "FAIL ×");

    // 释放资源
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled);  free(h_C_cpu);  // 新增！
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_naive); cudaFree(d_C_tiled);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}