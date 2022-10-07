1. 该文件夹放置了C++实现DNN的源码
2. 函数描述
    - dnn.cpp: 使用C++实现的DNN网络，包含三个全连接层
    - activations.h：实现激活函数ReLu, x -> max(0, x)
    - rain/weights.h: 存放全连接层的网络参数，rain表示数据集名
    - run_hls.tcl: vivado-hls脚本，用于生成Vivado HLS工程
    - run.tcl: vivado脚本，用于创建Vivado工程，进行综合仿真，生成资源占用与硬件功耗报告
    - script.py: 执行run_hls.tcl脚本，生成Vivado HLS工程，执行run.hls创建Vivado工程，进行综合仿真，生成资源占用与硬件功耗报告,执行report.py生成实验报告
3. 生成Vivado HLS和Vivado工程
    - `./script.sh`