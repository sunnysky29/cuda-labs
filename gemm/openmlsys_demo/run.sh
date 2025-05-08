#!/bin/bash
# 参考：8.4. 加速器实践
# https://openmlsys.github.io/chapter_accelerator/accelerator_practise.html

    #  --expt-relaxed-constexpr \    # 允许设备代码调用 constexpr 主机函数
    #  -Wno-deprecated-gpu-targets \  # 禁止旧架构弃用警告
    #  -Xcompiler -fopenmp \         # 启用 OpenMP 支持

# # 编译 CUDA 程序
# nvcc -I/usr/include/eigen3 \
#      --expt-relaxed-constexpr \
#      -Wno-deprecated-gpu-targets \
#      -Xcompiler -fopenmp \
#      -arch=sm_86 \
#      first_attempt.cu -o output

# # 检查编译是否成功
# if [ $? -eq 0 ]; then
#     echo "编译成功，正在运行程序..."
#     ./output  # 运行生成的可执行文件
# else
#     echo "编译失败，请检查错误！"
# fi

# --------------------------

# 编译 CUDA 程序
nvcc -I/usr/include/eigen3 \
     --expt-relaxed-constexpr \
     -Xcompiler -fopenmp \
     -arch=sm_86 \
     gem_use_smem.cu -o output

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "编译成功，正在运行程序..."
    ./output  # 运行生成的可执行文件
else
    echo "编译失败，请检查错误！"
fi