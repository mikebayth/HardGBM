import math
import os
import sys

from train_xgb import xgbTrain
import load_data
import callbacks as cb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
'''
    命令行参数：
'''


if len(sys.argv) < 5:
    print('need more params')
    exit()


dataset = sys.argv[1]
X,y,dic = load_data.load_by_argv(dataset)
X = X.values
y = y.values
y = y.flatten()


dic = sys.argv[2]

num_round = int(sys.argv[3])


'''
    使用OOitr_cb，需要设置保留 15% 20% 25% 30% 共4组不同约简规模备选
'''

callback=[
    # part1
    cb.origin_cb ,             # 0
    cb.odd_cb ,                # 2
    cb.even_cb,                # 3
    cb.odd_cb ,                # 4
    cb.even_cb,                # 5
    cb.odd_cb ,                # 6
    cb.even_cb,                # 7

    cb.kplus_odd_cb ,          # 8
    cb.kplus_even_cb ,         # 9
    cb.kplus_odd_cb,           # 10
    cb.kplus_even_cb,          # 11

    cb.OO_itr_cb,               # 15 3
    cb.OO_itr_cb,               # 20 3
    cb.OO_itr_cb,               # 25 3
    cb.OO_itr_cb,               # 15 5
    cb.OO_itr_cb,               # 20 5
    cb.OO_itr_cb,               # 25 5

    cb.OO_itr_cb,               # 15 3
    cb.OO_itr_cb,               # 20 3
    cb.OO_itr_cb,               # 25 3
    cb.OO_itr_cb,               # 15 5
    cb.OO_itr_cb,               # 20 5
    cb.OO_itr_cb,               # 25 5

]

#设置回调参数
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
arg=[
    [["origin", 1], num_round], # 0

    [["odd_0", 1], num_round, 0],   # 1
    [["even_0", 1], num_round, 0],  # 2
    [["odd_1", 1], num_round, 1],   # 3
    [["even_1", 1], num_round, 1],  # 4
    [["odd_3", 1], num_round, 3],   # 5
    [["even_3", 1], num_round, 3],  # 6

    [["k+1 odd_1", 1], 0, num_round, 1], # 7
    [["k+1 even_1", 1], 0, num_round, 1], # 8
    [["k+1 odd_3", 1], 0, num_round, 3],  # 9
    [["k+1 even_3", 1], 0, num_round, 3], # 10
    
    [["oo_15", 1], None, None, int(num_round*0.15), 3, int(num_round/2)],   # 11
    [["oo_20", 1], None, None, int(num_round*0.20), 3, int(num_round/2)],   # 12
    [["oo_25", 1], None, None, int(num_round*0.25), 3, int(num_round/2)],   # 13
    [["oo_15", 1], None, None, int(num_round*0.15), 5, int(num_round/3)],   # 14
    [["oo_20", 1], None, None, int(num_round*0.20), 5, int(num_round/2)],   # 15 
    [["oo_25", 1], None, None, int(num_round*0.25), 5, int(num_round/3)],  # 16

    [["oo_15_1", 1], None, None, int(num_round*0.15), 3, int(num_round*0.7)],   # 17
    [["oo_20_2", 1], None, None, int(num_round*0.20), 3, int(num_round*0.7)],   # 18
    [["oo_25_3", 1], None, None, int(num_round*0.25), 3, int(num_round*0.7)],   # 19
    [["oo_15_4", 1], None, None, int(num_round*0.15), 5, int(num_round)],   # 20
    [["oo_20_5", 1], None, None, int(num_round*0.20), 5, int(num_round)],   # 21
    [["oo_25_6", 1], None, None, int(num_round*0.25), 5, int(num_round)]  # 22
]

#设置循环内参数，这里指定 arg[1][1]是ytrain_mean
'''
    从上面的描述可以推测相应的arg_param设置方案(其实和lasso的差不多，不明白的话请参考work_dataset.py)
'''
arg_param=[
    ["X_train", 11, 1], ["y_train", 11, 2],
    ["X_train", 12, 1], ["y_train", 12, 2],
    ["X_train", 13, 1], ["y_train", 13, 2],
    ["X_train", 14, 1], ["y_train", 14, 2],
    ["X_train", 15, 1], ["y_train", 15, 2],
    ["X_train", 16, 1], ["y_train", 16, 2],
    ["X_train", 17, 1], ["y_train", 17, 2],
    ["X_train", 18, 1], ["y_train", 18, 2],
    ["X_train", 19, 1], ["y_train", 19, 2],
    ["X_train", 20, 1], ["y_train", 20, 2],
    ["X_train", 21, 1], ["y_train", 21, 2],
    ["X_train", 22, 1], ["y_train", 22, 2],
]


statistic_cb = [
    cb.acc_cb,
    cb.chosen_num_cb,
    cb.time_cb,
    cb.weight_cb,
]

#Xgboost参数
params = {
    'booster': 'gbtree',                # 基类学习器类型，可选gbtree,dart,linear

    'objective': 'binary:logistic',     # 决策树的损失函数， reg:linear 就是mse，binary:logistic,对数损失，binary:hinge,svm的损失函数，multi:softmax
    'subsample': 1,                     # 用train set的0.7训练
    'max_depth': 5,                     # 最大深度
    'eta': 0.05,                        # 衰减系数
    'gamma': 0,                       # 重要调参参数：分支界限（在一个叶子上，分裂后损失函数要达到此界限的收益才能进行），也是叶节点数量惩罚项的系数
    'alpha': 0,                         # L1正则项的系数，一般不用
    'lambda': 0.5,                        # L2正则项的系数，调参不好用
    'colsample_bytree': 1,              # 特征选取系数（和RF一样）
    'colsample_bylevel':1,
    'colsample_bynode':1,
    'min_child_weight': 1,  # 叶节点至少有3个数据才能分裂

    #'max_delta_step':0.4,               # 样本分布不均衡时用这两个可以调节，实际效果极差
    #'scale_pos_weight': 1,

    'silent': 1,  # 不打印输出
    'nthread': 32,       # 最大使用线程
}

params['max_depth'] = int(sys.argv[4])


info = {}
xgb_model_path = 'xgboost_models/'
xgb_output_path = 'xgboost_output/'
if os.path.exists(xgb_model_path) == False:
    os.mkdir(os.getcwd() + "/" + xgb_model_path)
if os.path.exists(xgb_output_path) == False:
    os.mkdir(os.getcwd() + "/" + xgb_output_path)

name = '{}_{}_{}/'.format(dataset,num_round,params['max_depth'])
info['model_path'] = xgb_model_path + name
info['output_path'] = xgb_output_path
info['dataset'] = sys.argv[1]
info['output_dic'] = dic

fold_num=10
xgbTrain(X, y, fold_num, params, \
    callback, arg, arg_param, \
        xgb_output_path + dic, num_round, statistic_cb, info)