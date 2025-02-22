#include<stdio.h>
#include<stdlib.h> 

__global__ void print_from_gpu(void) {
	printf("Hello World! from thread [%d,%d] \
		From device\n", threadIdx.x,blockIdx.x);

	#if defined(__CUDA_ARCH__)
	printf("Running on SM %d\n", __CUDA_ARCH__);   // 查看实际运行的 SM
	#endif
}

int main(void) { 
	printf("Hello World from host!\n"); 
	print_from_gpu<<<1,1>>>();
	cudaDeviceSynchronize();
return 0; 
}

