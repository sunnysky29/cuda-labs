/*
展示 block 输出的乱序

*/
#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1


__global__ void hello()
{
    printf("hello world! i am a thread  in block %d\n", blockIdx.x);
}

int main(int argc, char **argv){
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    cudaDeviceSynchronize();

    printf("that is all!\n");

    return 0;
}
