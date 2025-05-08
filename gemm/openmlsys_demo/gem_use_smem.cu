#include "util.cuh"
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <Eigen/Core>


namespace {
template <typename LayoutTile, typename LayoutBlock, typename LayoutThread>
__global__ void gemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
  constexpr unsigned ratio = sizeof(openmlsys::float4) / sizeof(float);
  using LayoutTileT =
      openmlsys::Layout<LayoutTile::m / ratio, LayoutTile::n / ratio,
                        LayoutTile::k / ratio>;
  using LayoutThreadT =
      openmlsys::Layout<LayoutThread::m / ratio, LayoutThread::n / ratio>;
  constexpr unsigned blockSize = LayoutBlock::m * LayoutBlock::n;
  constexpr openmlsys::float4 float4Zero{0.f, 0.f, 0.f, 0.f};

  __shared__ openmlsys::float4 tileA[LayoutTile::m][LayoutTileT::k];
  __shared__ openmlsys::float4 tileB[LayoutTile::k][LayoutTileT::n];

  const unsigned nInTileC = threadIdx.x % LayoutBlock::m;
  const unsigned mInTileC = threadIdx.x / LayoutBlock::m;

  const unsigned kInTileA = threadIdx.x % LayoutTileT::k;
  const unsigned mInTileA = threadIdx.x / LayoutTileT::k;

  const unsigned nInTileB = threadIdx.x % LayoutTileT::n;
  const unsigned kinTileB = threadIdx.x / LayoutTileT::n;

  openmlsys::Tensor2D<const openmlsys::float4> pA{A, M, K / ratio};
  pA.addOffset(LayoutTile::m * blockIdx.y + mInTileA, kInTileA);
  openmlsys::Tensor2D<const openmlsys::float4> pB{B, K, N / ratio};
  pB.addOffset(kinTileB,
               LayoutTileT::n * blockIdx.x + nInTileB * LayoutThreadT::n);
  openmlsys::Tensor2D<openmlsys::float4> pC{C, M, N / ratio};
  pC.addOffset(LayoutTile::m * blockIdx.y + mInTileC * LayoutThread::m,
               LayoutTileT::n * blockIdx.x + nInTileC * LayoutThreadT::n);

  constexpr unsigned tileSizeA = LayoutTile::m * LayoutTile::k;
  constexpr unsigned tileSizeB = LayoutTile::n * LayoutTile::k;
  constexpr unsigned tileIterationsA = tileSizeA / blockSize / ratio;
  constexpr unsigned tileGlobalIntervalA = blockSize / LayoutTileT::k;
  constexpr unsigned tileComputeIterationsA = LayoutTileT::m / LayoutBlock::m;
  constexpr unsigned tileSharedIntervalA =
      LayoutTile::m / tileComputeIterationsA;
  constexpr unsigned tileIterationsB = tileSizeB / blockSize / ratio;
  constexpr unsigned tileGlobalIntervalB = blockSize / LayoutTileT::n;
  constexpr unsigned tileComputeIterationsB = LayoutTileT::n / LayoutBlock::n;
  constexpr unsigned tileSharedIntervalBT =
      LayoutTileT::n / tileComputeIterationsB;

  openmlsys::float4 bufferA[tileIterationsA];
  openmlsys::float4 bufferB[tileIterationsB];
  bool validLoadTileA[tileIterationsA];
  bool validLoadTileB[tileIterationsB];

#pragma unroll
  for (unsigned i = 0; i < tileIterationsA; ++i) {
    validLoadTileA[i] = pA.validRowOffset(i * tileGlobalIntervalA);
  }

#pragma unroll
  for (unsigned i = 0; i < tileIterationsB; ++i) {
    validLoadTileB[i] = pB.validColOffset(0);
  }

  openmlsys::float4 c[tileComputeIterationsA * LayoutThread::m]
                     [tileComputeIterationsB * LayoutThreadT::n];
  memset(c, 0, sizeof(c));

  openmlsys::float4 fragmentA[tileComputeIterationsA * LayoutThreadT::m];
  openmlsys::float4 fragmentB[tileComputeIterationsB * LayoutThreadT::n];

  for (unsigned i = 0; i < K; i += LayoutTile::k) {
#pragma unroll
    for (unsigned j = 0; j < tileIterationsA; ++j) {
      validLoadTileA[j] &= pA.validColOffset(0);
      bufferA[j] =
          validLoadTileA[j] ? pA(j * tileGlobalIntervalA, 0) : float4Zero;
    }

#pragma unroll
    for (unsigned j = 0; j < tileIterationsB; ++j) {
      validLoadTileB[j] &= pB.validRowOffset(j * tileGlobalIntervalB);
      bufferB[j] =
          validLoadTileB[j] ? pB(j * tileGlobalIntervalB, 0) : float4Zero;
    }

    __syncthreads();
#pragma unroll
    for (unsigned a = 0; a < tileIterationsA; ++a) {
      tileA[mInTileA + a * tileGlobalIntervalA][kInTileA] = bufferA[a];
    }

#pragma unroll
    for (unsigned a = 0; a < tileIterationsB; ++a) {
      tileB[kinTileB + a * tileGlobalIntervalB][nInTileB] = bufferB[a];
    }
    __syncthreads();

#pragma unroll
    for (unsigned j = 0; j < LayoutTile::k; j++) {
#pragma unroll
      for (unsigned a = 0; a < tileComputeIterationsA; ++a) {
#pragma unroll
        for (unsigned b = 0; b < LayoutThread::m; ++b) {
          fragmentA[a][b] =
              tileA[a * tileSharedIntervalA + mInTileC * LayoutThread::m + b]
                   [j / ratio][j % ratio];
        }
      }
#pragma unroll
      for (unsigned a = 0; a < tileComputeIterationsB; ++a) {
        fragmentB[a] = tileB[j][a * tileSharedIntervalBT + nInTileC];
      }
#pragma unroll
      for (unsigned d = 0; d < tileComputeIterationsA * LayoutThread::m; ++d) {
#pragma unroll
        for (unsigned e = 0; e < tileComputeIterationsB * LayoutThreadT::n;
             ++e) {
          c[d][e] =
              c[d][e] + fragmentB[e] *
                            fragmentA[d / LayoutThread::m][d % LayoutThread::m];
        }
      }
    }
    pA.addOffset(0, LayoutTileT::k);
    pB.addOffset(LayoutTile::k, 0);
  }

#pragma unroll
  for (auto &a : c) {
#pragma unroll
    for (auto &b : a) {
      b = b * alpha;
    }
  }

#pragma unroll
  for (unsigned i = 0; i < tileComputeIterationsA; ++i) {
#pragma unroll
    for (unsigned a = 0; a < LayoutThread::m; a++) {
      const bool mValid = pC.validRowOffset(a);
#pragma unroll
      for (unsigned b = 0; b < tileComputeIterationsB; b++) {
        const bool nValid = pC.validColOffset(b * tileSharedIntervalBT);
        if (mValid && nValid) {
          openmlsys::float4 result{c[a + i * LayoutThread::m][b]};
          if (beta != 0) {
            result = result + pC(a, b * tileSharedIntervalBT) * beta;
          }
          pC(a, b * tileSharedIntervalBT) = result;
        }
      }
    }
    pC.addOffset(tileSharedIntervalA, 0);
  }
}
}  // namespace

void gemmUseSmem(const float *deviceAPtr, const float *deviceBPtr,
                 float *deviceCPtr, float alpha, float beta, unsigned M,
                 unsigned N, unsigned K) {
  using LayoutTile = openmlsys::Layout<128, 128, 16>;
  using LayoutBlock = openmlsys::Layout<16, 16>;
  using LayoutThread = openmlsys::Layout<4, 4>;

  dim3 block(LayoutBlock::m * LayoutBlock::n);
  dim3 grid((M - 1) / LayoutTile::m + 1, (N - 1) / LayoutTile::n + 1);

  gemmKernel<LayoutTile, LayoutBlock, LayoutThread><<<grid, block>>>(
      deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
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
    gemmUseSmem(deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
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