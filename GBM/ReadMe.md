1. Python code for ensemble prune method for gradient boosted trees
2. Function description
   - load_data.py: Perform data preprocessing, data cleaning, and load data for model training and testing. The datasets that are stored in `dataset\` directory come from the open source data set of UCI, Kaggle and data world.
   - train_xgb.py: Train XGBoost and test the accuracy using ten-fold cross-validation, then save the trained model, training time, accuracy and standard deviation 
   - train_lgb.py: Train LightGBM and test the accuracy using ten-fold cross-validation, then save the trained model, training time, accuracy and standard deviation 
   - prune_xgb.py: Use `train_xgb.py` to train XGBoost and use the SHR and SAR methods in `callback.py` to prune XGBoost, and record the accuracy of the pruned model, running time, the pruned model, etc.
   - prune_lgb.py: Use `train_lgb.py` to train LightGBM and use the SHR and SAR methods in `callback.py` to prune LightGBM, and record the accuracy of the pruned model, running time, the pruned model, etc.
   - callback.py: The file holds callback functions of the SHR and SAR ensemble prune methods called in `prune_xgb.py` and `prune_lgb.py`

3. Python libraries
   - lightgbm - version 3.3.2
   - matplotlib - version 3.4.3
   - numpy - version 1.21.2
   - pandas - version 1.3.4
   - progress - version 1.6
   - scikit_learn - version 1.1.2
   - scipy - version 1.7.1
   - xgboost - version 1.4.2
4. How to build HardGBM project
Taking the `car` dataset as an example, follow the steps below to get the pruned model,then you can switch the working directory to `hw_impl/` to generate a vivado-based hardware project
   - `python prune_xgb.py car car4 200 10`
      + Input parameter description:
         * `car` &nbsp;#  dataset name
         * `car4`  &nbsp;# epochs corresponding to different hyperparameters
         * `200`  &nbsp;# number of decision trees
         * `10`  &nbsp;# maximum depth of decision trees
      + Expected output
         * trained XGBoost model - after ten-fold cross-validation, the model corresponding to each fold is saved in the folder `xgboost_models/`
         * pruned model - use the SHR and SAR methods to prune XGBoost model, save pruned model information in the file `xgboost_ouput/weight.csv`.Different decision trees have different weights, and a weight of 0 indicates a reduced decision tree.
         * Model performance - the performance of XGboost model and pruned model are recorded in the following files, `xgboost_ouput/acc.csv` for the average accuracy of the model, `xgboost_ouput/acc_std.csv` for the variance of the model's accuracy, time.csv for model runtime, `xgboost_ouput/tree_num.csv` for the number of decision trees
      + Purpose of the experiment  
      To get trained XGBoost model and pruned model, furthermore compare their performance.
   - `python prune_lgb.py car car2 200 4`
       + Input parameter description:
         * `car` &nbsp;#  dataset name
         * `car4`  &nbsp;# epochs corresponding to different hyperparameters
         * `200`  &nbsp;# number of decision trees
         * `4`  &nbsp;# maximum depth of decision trees
      + Expected output
         * trained LightGBM model - after ten-fold cross-validation, the model corresponding to each fold is saved in the folder `lgb_models/`
         * pruned model - use the SHR and SAR methods to prune LightGBM model, save pruned model information in the file `lgb_ouput/weight.csv`.Different decision trees have different weights, and a weight of 0 indicates a reduced decision tree.
         * Model performance - the performance of LightGBM model and pruned model are recorded in the following files, `lgb_ouput/acc.csv` for the average accuracy of the model, `lgb_ouput/acc_std.csv` for the variance of the model's accuracy, time.csv for model runtime, `lgb_ouput/tree_num.csv` for the number of decision trees
      + Purpose of the experiment  
      To get trained LightGBM model and pruned model, furthermore compare their performance.