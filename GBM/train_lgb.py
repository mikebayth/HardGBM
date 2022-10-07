# -*- coding: utf-8 -*-
# @Author  : LongyunBian
# @Time    : 2022/5/24 16:13
import os
import pickle

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def train(dtrain, dtest, y_test, params, output_path, callback, arg, num_rounds, model):
    """
    :param dtrain: 包括了训练数据和训练label
    :param dtest:  测试数据
    :param y_test: 测试label
    :param params:    lgboost参数
    :param output_path: 输出路径
    :param callback:   回调函数表
    :param arg:        回调函数参数
    :param num_rounds:  训练轮次
    :return:        返回回调函数可能产生的输出
    """

    f = open(output_path, "w")

    # 第i颗树的测试样本的输出 output[0]的输出就是0
    output = []
    output.append([])
    for i in range(len(y_test)):
        output[0].append(0)

    tmp_ans = np.zeros(len(output[0]))
    for i in range(1, num_rounds + 1):
        # output.append([])
        ans = model.predict(dtest, num_iteration=i, raw_score=True)
        # for xxx in range(len(ans)):
        #     ans[xxx] = np.log(ans[xxx] / (1 - ans[xxx]))
        output.append(ans - tmp_ans)
        tmp_ans = ans
    output[0] = np.array(output[0])
    results = []
    stats = []

    for i in range(len(callback)):
        # 调用那么多个回调函数
        cb_out, stat = callback[i](output, arg[i], f, y_test, [model, num_rounds])
        stats.append(stat)
        if (arg[i][0][1] != 0):
            results.append(cb_out)

    counter = 0
    f.write("\n")
    for i in range(len(arg)):
        if arg[i][0][1] != 0:
            f.write("{:<9}\t\t".format(arg[i][0][0]))
            counter += 1

    if (counter == 0):
        f.close()
        os.remove(output_path)
        return stats

    y_test = y_test.flatten()

    f.write("label\n")
    for j in range(len(y_test)):
        for i in range(len(results)):
            if arg[i][0][1] != 0:
                f.write("{:<9.6}\t\t".format(results[i][j]))
        f.write("{:<9.6}\n".format(y_test[j].astype('float64')))

    f.close()

    return stats


def lgbTrain(X, y, kfold, params, callback, arg, arg_param, path, num_round, statistic_cb, info):
    """
    :param X: 输入数据部分
    :param y: label部分
    :param kfold: fold数
    :param params: lgboost参数表
    :param callback: 回调函数
    :param arg: 回调函数需要的参数
    :param arg_param: 不能直接指定，需要在框架中计算的需要的参数
    :param path: 输出路径
    :param num_round: 训练轮次
    :return: 无
    """

    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=7)
    fold_id = 0
    rets = []

    if os.path.exists(info['model_path']) == False:
        os.mkdir(os.getcwd() + "/" + info['model_path'])
    if os.path.exists(path) == False:
        os.mkdir(os.getcwd() + "/" + path)

    for train_index, test_index in kf.split(X, y):

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # dtest = lgb.Dataset(X_test)
        dtest = X_test
        dtrain = lgb.Dataset(X_train, label=y_train)

        file_path = '{}{}_fold{}.dat'.format(info['model_path'], info['dataset'], fold_id)
        if os.path.exists(file_path) == False:
            model = lgb.train(params, dtrain, num_round)
            pickle.dump(model, open(file_path, "wb"))
        else:
            model = pickle.load(open(file_path, "rb"))

        # traindata_output = []
        # traindata_output.append([])
        # for i in range(len(y_train)):
        #     traindata_output[0].append(0)
        #
        # tmp_ans = np.zeros(len(traindata_output[0]))
        # for i in range(1, num_round + 1):
        #     traindata_output.append([])
        #     ans = model.predict(X_train, num_iteration=i, raw_score=True)
        #     traindata_output.append(ans - tmp_ans)
        #     # for j in range(len(ans)):
        #     #     traindata_output[i].append(ans[j] - tmp_ans[j])
        #
        #     tmp_ans = ans

        for i in range(len(arg_param)):
            if arg_param[i][0] == "y_train_mean":
                mean_tar = np.array(y_train).mean()
                arg[arg_param[i][1]][arg_param[i][2]] = mean_tar
            elif arg_param[i][0] == "fold_id":
                arg[arg_param[i][1]][arg_param[i][2]] = fold_id
            elif arg_param[i][0] == "y_train":
                arg[arg_param[i][1]][arg_param[i][2]] = y_train
            elif arg_param[i][0] == "X_train":
                arg[arg_param[i][1]][arg_param[i][2]] = X_train
            # elif arg_param[i][0] == "Prediction_train":
            #     arg[arg_param[i][1]][arg_param[i][2]] = traindata_output
            else:
                continue

        # k折交叉验证，Train那么多次。
        ret = train(dtrain, dtest, y_test, params,
                    path + "/file{}.txt".format(fold_id),
                    callback=callback, arg=arg,
                    num_rounds=num_round, model=model)

        rets.append(ret)
        fold_id += 1

    for k in range(len(statistic_cb)):
        statistic_cb[k](rets, path, info)
