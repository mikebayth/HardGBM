# Hardware comparison experiment
1. Function description
    - dump_tree.py: Save the XGBoost gradient boosting tree structure description file
    - get_verilog.py: Call the hardware code construction function in the package `utilts` to generate the hardware code of the gradient boosting tree
    - get_verilog_bucket.py: Call the hardware code construction function in the package `utilts`, and use the bucketing method to reduce resource overhead and power consumption to generate the hardware code of the gradient boosting tree
    - run.tcl: The vivado script used to create a viavado project, perform comprehensive simulation, and generate resource usage and hardware power consumption reports
    - run_vivado.py: Batch run vivado scripts
    - report.py: Read the vivado resource overhead and hardware power consumption report, generate the experimental report for before and after reduction, and use the bucket method
2. The folder `utils` places the source code for reading the tree structure, generating the hardware code
    - Tree_Reader.py: According to the XGBoost gradient boosting tree tree structure description file, read the tree used to build the hardware code
    - LGB_Tree_Reader.py: Read the tree used to build the hardware code according to the description gradient boosting tree structure API provided by LightGBM
    - Build_vlg.py: Build hardware code based on the tree read
    - combine_leaves.py: Bucket method implementation 
3. Software used:   
    OS:  Ubuntu 18.04.5 LTS 
    Python 3.8.12
    Vivado 2018.3 
4. Taking the car dataset as an example, follow the steps below to generate and generate hardware engineering and simulation experiment reports
    - `./dump_tree.sh`
        + Input - Trained XGBoost models in the folder `../xgboost_models/` 
        + Output -  XGBoost gradient boosting tree structure description files dumped in the floder `dump_tree/`
        + Purpose  - Get the decision tree structure for automatic hardware code generation
    - `./get_verilog.sh`
        + Input - XGBoost gradient boosting tree structure description files dumped in the floder `dump_tree/` or the description gradient boosting tree structure API provided by LightGBM
        + Output  - The hardware code of the gradient boosting tree
    - `./get_verilog_bucket.sh`
        + Input - XGBoost gradient boosting tree structure description files dumped in the floder `dump_tree/` or the description gradient boosting tree structure API provided by LightGBM
        + Output - The hardware code of the gradient boosting tree using the bucketing method
    - `python run_vivado.py`
        + Input - The hardware code of the gradient boosting tree
        + Output - Vivado project with resource overhead and hardware power consumption report
    - `python report.py`
        + Input - Vivado resource overhead and hardware power consumption report
        + Output - The files `report_lgb.txt`,`report_lgb.txt`,`report_xgb_bucket` recorded the value of `LUTS`, `FFs`, `Power`
5. Hardware report
    - place_power.rpt：Power consumption reporting for hardware project
    - place_utilization.rpt: Overhead reporting for hardware project 
    - The floder `hardware_xgb/` stores XGBoost and SAR hardware reports respectively and the floder `hardware_lgb/` stores LightGBM and SAR hardware reports respectively
    - Directory structure
    
    ```
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
    ```
