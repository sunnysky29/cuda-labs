CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}  # ccbin 指定主机端编译器
TARGET=hello_world

INCLUDES= -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo #-m64：编译为 64 位程序; -lineinfo：在 GPU 代码中包含调试信息

IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
# 如果 CUDA 版本 < 11.0，支持的计算能力（SM）是 35, 37, 50, 52, 60, 61, 70, 75
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 80 86 90
endif

# arch=compute_xx, 代表 虚拟架构（Virtual Architecture），
# 					即 PTX（Parallel Thread Execution）中间代码的版本。
# code=sm_xx, 代表 物理架构（Real Architecture），
#			 即生成的 GPU 机器代码。
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

hello_world: hello_world.cu
	@echo "Compiling with GENCODE_FLAGS: ${GENCODE_FLAGS}"
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<

clean:
	rm -f ${TARGET}
