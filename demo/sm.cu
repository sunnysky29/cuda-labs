// 检测系统中所有可用的 CUDA GPU 设备，并输出每个设备的详细信息，
// 例如设备名称和其支持的计算能力（Compute Capability）
// Device 0: NVIDIA GeForce RTX 3070 Ti Laptop GPU
// Compute Capability: 8.6


#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    }

    return 0;
}
