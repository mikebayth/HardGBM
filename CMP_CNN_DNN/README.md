This is the CNN/DNN comparison codes to our resemble learning method.

1. It contains the following directories:
    - cnn-on-fpga/ Vivado-based CNN FPGA project, a customized versions of the repository https://github.com/MasLiang/CNN-On-FPGA. We modified the project to fit our CNN architecture and tasks. 
    - dnn-using-hls/ Vivado-based DNN FPGA project, a customized versions of the repository https://github.com/amiq-consulting/CNN-using-HLS. We removed the convolutional and pooling layers and modified the input and output layers to fit our network architecture and task.
    - py/ python codes for training and testing CNN/DNN for the same datasets. The datasets mainly come from UCI/Kaggle. Moreover, we offer a quantized method to store weights and inputs to arbitrary fixed-points.
2. Software used:
    OS:  Ubuntu 18.04.5 LTS 
    Vivado HLS 2018.3 - Simulation results and Synthesis
    Vivado 2018.3 - Report the utilization and power
    Python libraries
    - numpy - version 1.21.2
    - pandas - version 1.3.4
    - scikit_learn - version 1.1.2
    - scipy - version 1.7.1
    - torch - version 1.9.0





