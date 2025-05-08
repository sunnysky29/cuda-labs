/*

以看到，矩阵C中各个元素的计算是独立的。可以利用GPU的大量线程去分别计算矩阵C
 中相应的元素，以达到并行计算的目的，
*/


#include <omp.h>
#include <Eigen/Core>
#include <ctime>
#include <iostream>

__global__ void gemmKernel(const float *A, const float *B, float *C,
                          float alpha, float beta, unsigned M, unsigned N,
                          unsigned K) {
    unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
    if (m >= M || n >= N) return;
    float c = 0;
    for (unsigned k = 0; k < K; ++k) {
        c += A[m * K + k] * B[k * N + n];
    }
    c = c * alpha;
    float result = c;
    if (beta != 0) {
        result = result + C[m * N + n] * beta;
    }
    C[m * N + n] = result;
}

void gemmNaive(const float *A, const float *B, float *C, float alpha,
               float beta, unsigned M, unsigned N, unsigned K) {
    // dim3 block(32, 32);
    dim3 block(16,16);

    dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
    gemmKernel<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}

int main() {
    // GPU信息输出
    int gpu_rank = 0;
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, gpu_rank);
    cudaSetDevice(gpu_rank);
    printf("GPU %s status: ", deviceProp.name);
    double boostFrequency = deviceProp.clockRate / 1e6;
    int fp32CoresNum = 640;
    double peakPerformance = boostFrequency * fp32CoresNum * 2;
    printf("clock rate %.3f GHz, FP32 cores num %d, FP32 peak throughput %.3f GFLOPS\n",
           boostFrequency, fp32CoresNum, peakPerformance);

    // 设置矩阵大小和参数
    omp_set_num_threads(omp_get_num_procs());
    unsigned M = 1024, N = 1024, K = 1024;
    float alpha = 1., beta = 0.;
    
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // 使用当前时间作为种子

    // 分配设备内存
    float *deviceAPtr, *deviceBPtr, *deviceCPtr;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A{M, K}, B{K, N}, C{M, N};
    
    // 初始化随机矩阵
    A.setRandom();
    B.setRandom();
    C.setRandom();
    
    // 打印第一个元素作为示例
    std::cout << "Matrix A first element: " << A(0, 0) << std::endl;
    std::cout << "Matrix B first element: " << B(0, 0) << std::endl;
    std::cout << "Matrix C first element: " << C(0, 0) << std::endl;

    // 拷贝数据到设备
    cudaMalloc(&deviceAPtr, M * K * sizeof(float));
    cudaMemcpy(deviceAPtr, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&deviceBPtr, K * N * sizeof(float));
    cudaMemcpy(deviceBPtr, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&deviceCPtr, M * N * sizeof(float));
    cudaMemcpy(deviceCPtr, C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // 计时GPU计算
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
    gemmNaive(deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    printf("GPU time: %.3f ms\n", milliseconds);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);

    // CPU计算
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hostResult{M, N}, deviceResult{M, N};
    clock_t begin, end;
    begin = clock();
    hostResult = alpha * (A * B) + beta * C;
    end = clock();
    printf("CPU time: %.3f ms\n", double(end - begin) / CLOCKS_PER_SEC * 1e3);

    // 拷贝结果回主机
    cudaMemcpy(deviceResult.data(), deviceCPtr, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // 打印第一个结果元素
    std::cout << "CPU result first element: " << hostResult(0, 0) << std::endl;
    std::cout << "GPU result first element: " << deviceResult(0, 0) << std::endl;

    // 计算误差
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray = (hostResult - deviceResult).array().abs();
    printf("Max Error: %f\n", diffArray.maxCoeff());

    // 计算GFLOPS
    double GFLOPS = 2 * 1e-9 * M * N * K / (milliseconds * 1e-3);
    printf("GPU Throughput: %.3f GFLOPS\n", GFLOPS);

    // 释放设备内存
    cudaFree(deviceAPtr);
    cudaFree(deviceBPtr);
    cudaFree(deviceCPtr);

    return 0;
}