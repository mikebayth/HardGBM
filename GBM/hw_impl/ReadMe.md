1. 对于XGBoost与LightGBM约简前后的硬件对比实验
2. 函数描述
    - dump_tree.py: 保存XGBoost梯度提升树树结构描述文件，
    - get_verilog.py: 调用utilts包中的硬件代码构建函数，生成梯度提升树的硬件代码
    - get_verilog_bucket.py: 调用utilts包中的硬件代码构建函数，并使用用于降低资源开销与功耗的分桶方法，生成梯度提升树的硬件代码
    - run.tcl: vivado脚本，用于创建viavado工程，进行综合仿真，生成资源占用与硬件功耗报告
    - run_vivado.py: 批量运行vivado脚本
    - report.py: 读取vivado资源占用与硬件功耗报告，生成约简前后以及使用分桶方法实验报告
3. 文件夹 utils放置用于读取树结构，生成硬件代码的源代码
    - Tree_Reader.py: 根据XGBoost梯度提升树树结构描述文件，读取用于构建硬件代码的树
    - LGB_Tree_Reader.py: 根据LightGBM提供的描述梯度提升树树结构API，读取用于构建硬件代码的树
    - Build_vlg.py: 根据读取到的树，构建硬件代码
    - combine_leaves.py: 分桶方法实现 
4.  Software used:   
    OS:  Ubuntu 18.04.5 LTS 
    Python 3.8.12
    Vivado 2018.3 
5. 以car数据集为例,按照一下步骤生成生成硬件工程和仿真实验报告
    - `./dump_tree.sh`
    - `./get_verilog.sh`
    - `./get_verilog_bucket.sh`
    - `python run_vivado.py`
    - `python report.py`
6. 硬件报告
    - place_power.rpt：硬件工程的功耗报告
    - place_utilization.rpt: 硬件工程的开销报告
    - 目录结构 `hardware_xgb/`存放着分别存放XGBoost与SAR硬件报告 `hardware_lgb/`存放着分别存放LightGBM与SAR硬件报告 
    hardware_xgb/
    ├── car4_OO_fold0
    │   └── vivado_out
    │       ├── place_power.rpt 
    │       └── place_utilization.rpt
    └── car4_XGB_fold0
        └── vivado_out
            ├── place_power.rpt
            └── place_utilization.rpt

    hardware_lgb/
    ├── car2_OO_fold0
    │   └── vivado_out
    │       ├── place_power.rpt
    │       └── place_utilization.rpt
    └── car2_LGB_fold0
        └── vivado_out
            ├── place_power.rpt
            └── place_utilization.rpt

    