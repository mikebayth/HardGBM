# -*- coding: utf-8 -*-
# @Author  : LongyunBian
# @Time    : 2022/5/24 16:30
# import math
import os
import sys

from train_lgb import lgbTrain
import load_data
import callbacks as cb
# import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
'''
    命令行参数：
'''

if len(sys.argv) < 5:
    print('need more params')
    exit()

dataset = sys.argv[1]
X, y, dic = load_data.load_by_argv(dataset)
X = X.values
y = y.values
y = y.flatten()

dic = sys.argv[2]

num_round = int(sys.argv[3])

'''
    使用OOitr_cb，需要设置保留 15% 20% 25% 30% 共4组不同约简规模备选
'''

callback = [
    # part1
    cb.origin_cb,  # 0

    cb.OO_itr_l_cb,
    cb.OO_itr_l_cb,
    cb.OO_itr_l_cb,

    cb.OO_itr_l_cb,
    cb.OO_itr_l_cb,
    cb.OO_itr_l_cb,

    cb.OO_itr_l_cb,
    cb.OO_itr_l_cb,
    cb.OO_itr_l_cb,


]

# 设置回调参数
"""
    第一项是数组，专门放设置，目前有三项
        其中第一个是名字，在输出具体预测结果的表格中使用（如果不看这个表格等于没有用）
        第二个是要不要将具体内容输出到表格
"""
threshold = 0.2
retain = 0

'''
    OOitr的设置
    [[name,1], None(train data), None(train label), retain, base, used]
    OOscale的设置
    [[name,1], None(train data), None(train label), retain, base, used, GroupSize, MaxCOS]
    (以上凭借记忆书写，应该是对的，如果不对，请参照callbacks.py里相应的回调函数设置)
'''
arg = [
    [["origin", 1], num_round],  # 0

    [["oo_15", 1], None, None, int(num_round * 0.15), 3, int(num_round / 2)],  # 1
    [["oo_20", 1], None, None, int(num_round * 0.20), 3, int(num_round / 2)],  # 2
    [["oo_25", 1], None, None, int(num_round * 0.25), 3, int(num_round / 2)],  # 3

    [["oo_15", 1], None, None, int(num_round * 0.15), 15, int(num_round * 0.3)],  # 4
    [["oo_20", 1], None, None, int(num_round * 0.20), 15, int(num_round * 0.3)],  # 5
    [["oo_25", 1], None, None, int(num_round * 0.25), 15, int(num_round * 0.3)],  # 6

    [["oo_15", 1], None, None, int(num_round * 0.15), 20, int(num_round * 0.4)],  # 7
    [["oo_20", 1], None, None, int(num_round * 0.20), 20, int(num_round * 0.4)],  # 8
    [["oo_25", 1], None, None, int(num_round * 0.25), 20, int(num_round * 0.4)],  # 9



    # [["oo_15_1", 1], None, None, int(num_round * 0.15), 3, int(num_round * 0.7)],  # 7
    # [["oo_20_2", 1], None, None, int(num_round * 0.20), 3, int(num_round * 0.7)],  # 8
    # [["oo_25_3", 1], None, None, int(num_round * 0.25), 3, int(num_round * 0.7)],  # 9
]

# 设置循环内参数，这里指定 arg[1][1]是ytrain_mean
'''
    从上面的描述可以推测相应的arg_param设置方案(其实和lasso的差不多，不明白的话请参考work_dataset.py)
'''
arg_param = [
    ["X_train", 1, 1], ["y_train", 1, 2],
    ["X_train", 2, 1], ["y_train", 2, 2],
    ["X_train", 3, 1], ["y_train", 3, 2],
    ["X_train", 4, 1], ["y_train", 4, 2],
    ["X_train", 5, 1], ["y_train", 5, 2],
    ["X_train", 6, 1], ["y_train", 6, 2],
    ["X_train", 7, 1], ["y_train", 7, 2],
    ["X_train", 8, 1], ["y_train", 8, 2],
    ["X_train", 9, 1], ["y_train", 9, 2],

]

statistic_cb = [
    cb.acc_cb,
    cb.chosen_num_cb,
    cb.time_cb,
    cb.weight_cb,
]

# lgboost参数
params = {
    'learning_rate': 0.1,
    'objective': 'binary',
    'max_depth': 4,  # 最大深度
    'lambda_l1': 0.1,
     'lambda_l2': 0.2,
    'num_iterations': 200,
    'min_data_in_leaf' :5,
    'verbose': -1
    # 决策树的损失函数， reg:linear 就是mse，binary:logistic,对数损失，binary:hinge,svm的损失函数，multi:softmax
    # 'subsample': 1,  # 用train set的0.7训练
    # 'eta': 0.05,  # 衰减系数
    # 'gamma': 0,  # 重要调参参数：分支界限（在一个叶子上，分裂后损失函数要达到此界限的收益才能进行），也是叶节点数量惩罚项的系数
    # 'colsample_bytree': 1,  # 特征选取系数（和RF一样）
    # 'colsample_bylevel': 1,
    # 'colsample_bynode': 1,
    # 'min_child_weight': 1,  # 叶节点至少有3个数据才能分裂
    
    # 'silent': 1,  # 不打印输出
    # 'nthread': 32,  # 最大使用线程
}

params['max_depth'] = int(sys.argv[4])
params['num_iterations'] = num_round
info = {}
lgb_model_path = 'lgb_models/'
lgb_output_path = 'lgb_output/'
if os.path.exists(lgb_model_path) == False:
    os.mkdir(os.getcwd() + "/" + lgb_model_path)
if os.path.exists(lgb_output_path) == False:
    os.mkdir(os.getcwd() + "/" + lgb_output_path)

name = '{}_{}/'.format(dataset, num_round)
info['model_path'] = lgb_model_path + name
info['output_path'] = lgb_output_path
info['dataset'] = sys.argv[1]
info['output_dic'] = dic

fold_num = 10
lgbTrain(X, y, fold_num, params,
         callback, arg, arg_param,
         lgb_output_path + dic, num_round, statistic_cb, info)