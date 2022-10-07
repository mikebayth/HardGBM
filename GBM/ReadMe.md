1. Python code for ensemble prune method for gradient boosted trees
2. Function description
   - load_data.py: Perform data preprocessing, data cleaning, and load data for model training and testing. The datasets that are stored in `dataset\` directory come from the open source data set of UCI and Kaggle 
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
4. Taking the `car` dataset as an example, follow the steps below to get the pruned model,then you can switch the working directory to `hw_impl/` to generate a vivado-based hardware project
   - `./prune_xgb.sh`   
   - `./prune_lgb.sh`


