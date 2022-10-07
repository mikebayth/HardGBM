import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV,LinearRegression
from time import process_time, time
import math
from copy import deepcopy
plt.switch_backend('Agg')

def get_residual(x,y):
    sum=0
    for i in range(len(y)):
        diff = y[i]-x[i]
        if diff < 0:
            diff = -diff
        sum += diff
    return sum/len(y)

def get_acc(x,y):
    sum=0
    for i in range(len(y)):
        if x[i]>=0 and y[i]==1 or x[i]<0 and y[i]==0:
            sum+=1
    return sum/len(y)

def swap(arr,i,j):
    if i==j:
        return
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp

def partition(arr,pos,begin,end):
    key = arr[begin]
    l = begin
    r = begin+1
    while r < end:
        if arr[r] >= key:
            l += 1
            swap(arr,l,r)
            swap(pos,l,r)
        r += 1
    swap(arr,l,begin)
    swap(pos,l,begin)
    return l

def q_sort(arr,pos,begin,end):
    if begin < end:
        split = partition(arr,pos,begin,end)
        q_sort(arr,pos,begin,split)
        q_sort(arr,pos,split+1,end)

def find_topk(arr,topk):
    length = len(arr)
    arrtemp = deepcopy(arr)
    pos = np.array(range(length)).tolist()
    mask = np.zeros(length,dtype=np.int).tolist()
    q_sort(arrtemp,pos,0,length)
    position = pos[0:topk]
    for i in position:
        mask[i] = 1
    return mask, position


def partition_min(arr,pos,begin,end):
    key = arr[begin]
    l = begin
    r = begin+1
    while r < end:
        if arr[r] <= key:
            l += 1
            swap(arr,l,r)
            swap(pos,l,r)
        r += 1
    swap(arr,l,begin)
    swap(pos,l,begin)
    return l

def q_sort_min(arr,pos,begin,end):
    if begin < end:
        split = partition_min(arr,pos,begin,end)
        q_sort_min(arr,pos,begin,split)
        q_sort_min(arr,pos,split+1,end)

def find_mink(arr,topk):
    length = len(arr)
    arrtemp = deepcopy(arr)
    pos = np.array(range(length)).tolist()
    mask = np.zeros(length,dtype=np.int).tolist()
    q_sort_min(arrtemp,pos,0,length)
    position = pos[0:topk]
    for i in position:
        mask[i] = 1
    return mask, position


def find_max_cos(X,y,mask):
    ## 20220516 bly###
    vs = np.array(X).T
    v1 = y
    num = np.dot(v1, vs)  # 向量点乘
    modulo = np.linalg.norm(v1) * np.linalg.norm(vs, axis=0)  # 求模长的乘积
    res = num / modulo
    candidate_indexs = np.argsort(res)[::-1]
    for i in candidate_indexs:
        if(mask[i]==0):
            ans_index = i
            mask[ans_index]=1
            break
    ans_cos = res[ans_index]
    y = y - vs[:,ans_index]

    return ans_cos, ans_index, y
def OO_itr_choose(arg,model_info):
    X_train = arg[1]
    dtrain = xgb.DMatrix(X_train)
    y_train = deepcopy(arg[2])
    y_train = y_train.astype('float64')
    y_train[y_train==0] = -1
    y_train[y_train==1] = 1

    base = arg[4]   # 固定选择前base棵树。
    used = arg[5]   # 在前used棵树中做集成约减
    retain = arg[3] - base
    if used == 0:
        used = model_info[1]

    start_time = process_time()
    # 用train data 和 model 得到xgb预测的输出结果
    traindata_output = []
    traindata_output.append([])
    for i in range(len(y_train)):
        traindata_output[0].append(0)

    tmp_ans = traindata_output[0]

    for i in range(1, used + 1):
        traindata_output.append([])
        ans = model_info[0].predict(dtrain, ntree_limit=i, output_margin=True)
        # for xxx in range(len(ans)) :
        #     ans[xxx] = np.log(ans[xxx]/(1-ans[xxx]))
        for j in range(len(ans)):
            traindata_output[i].append(ans[j] - tmp_ans[j])
        tmp_ans = ans
    end_time = process_time()
    data_process_time = end_time - start_time

    
    # 用来算夹角的树
    traindata_for_OO = traindata_output[base+1:]

    # 在label中减去前base棵树的输出
    if base != 0:
        for i in range(len(y_train)):
            base_num = 0
            for j in range(1,base+1):
                base_num += traindata_output[j][i]
            y_train[i] -= base_num

    length = len(traindata_for_OO)
    mask = np.zeros(length,dtype=np.int).tolist()
    # 在这里加上kmeans逻辑
    # y_train, mask = kmeans_reduction(traindata_for_OO,y_train,mask)
    
    for k in range(retain):
        cos, index, new_y = find_max_cos(traindata_for_OO, y_train, mask)
        y_train = new_y

    return mask, data_process_time


def OO_itr_l_choose(arg, model_info):
    X_train = arg[1]
    # dtrain = xgb.DMatrix(X_train)
    y_train = deepcopy(arg[2])
    y_train = y_train.astype('float64')
    y_train[y_train == 0] = -1
    y_train[y_train == 1] = 1

    base = arg[4]  # 固定选择前base棵树。
    used = arg[5]  # 在前used棵树中做集成约减
    retain = arg[3] - base
    if used == 0:
        used = model_info[1]

    start_time = process_time()
    # 用train data 和 model 得到xgb预测的输出结果
    traindata_output = []
    traindata_output.append([])
    for i in range(len(y_train)):
        traindata_output[0].append(0)

    tmp_ans = traindata_output[0]

    for i in range(1, used + 1):
        traindata_output.append([])
        ans = model_info[0].predict(X_train, num_iteration=i, raw_score=True)
        # for xxx in range(len(ans)) :
        #     ans[xxx] = np.log(ans[xxx]/(1-ans[xxx]))
        for j in range(len(ans)):
            traindata_output[i].append(ans[j] - tmp_ans[j])
        tmp_ans = ans
    end_time = process_time()
    data_process_time = end_time - start_time

    # 用来算夹角的树
    traindata_for_OO = traindata_output[base + 1:]

    # 在label中减去前base棵树的输出
    if base != 0:
        for i in range(len(y_train)):
            base_num = 0
            for j in range(1, base + 1):
                base_num += traindata_output[j][i]
            y_train[i] -= base_num

    length = len(traindata_for_OO)
    mask = np.zeros(length, dtype=np.int).tolist()
    # 在这里加上kmeans逻辑
    # y_train, mask = kmeans_reduction(traindata_for_OO,y_train,mask)

    for k in range(retain):
        cos, index, new_y = find_max_cos(traindata_for_OO, y_train, mask)
        y_train = new_y

    return mask, data_process_time

def OO_itr_l_cb(output, arg, file, y_test, model_info):
    start_time = process_time()
    base = arg[4]
    used = arg[5]
    retain = arg[3] - base
    # 通过对树的初步选择返回一个mask，mask就是这棵树到底选不选。
    mask, data_process_time = OO_itr_l_choose(arg, model_info)

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i] == 1:
            weight.append(1)
        else:
            weight.append(0)

    OOitr = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            OOitr += np.array(output[i]) * weight[i]
    end_time = process_time()

    acc = get_acc(OOitr, y_test)
    mse = mean_squared_error(OOitr, y_test)
    file.write("OOitr_{}_{}_{} : mse={} ,acc={}\n".format(retain + base, used, base, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(retain + base, used))
    return OOitr, {'name': 'OOitr_{}_{}_{}'.format(retain + base, used, base), \
                   'acc': acc, 'mse': mse, 'chosen_num': retain + base, \
                   'total_num': used, 'time': end_time - start_time - data_process_time, \
                   'weight': weight[1:]}

def OO_itr_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    base = arg[4]
    used = arg[5]
    retain = arg[3] - base
    # 通过对树的初步选择返回一个mask，mask就是这棵树到底选不选。
    mask, data_process_time = OO_itr_choose(arg,model_info)

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i]==1:
            weight.append(1)
        else:
            weight.append(0)
    
    OOitr = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            OOitr += np.array(output[i]) * weight[i]
    end_time = process_time()

    acc = get_acc(OOitr, y_test)
    mse = mean_squared_error(OOitr, y_test)   
    file.write("OOitr_{}_{}_{} : mse={} ,acc={}\n".format(retain+base, used, base, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(retain+base,used))
    return OOitr, {'name': 'OOitr_{}_{}_{}'.format(retain+base, used, base), \
        'acc': acc, 'mse': mse, 'chosen_num':retain + base, \
            'total_num':used,'time':end_time-start_time-data_process_time,\
                'weight':weight[1:]}


#===============================================================

def find_scale_cos(X,y,mask,w, arg):
    topk = arg[6]
    mincos = arg[7]
    cos = []
    mody = np.linalg.norm(y)
    for i,x in enumerate(X):
        if mask[i] == 0:
            res = np.dot(x,y)/(np.linalg.norm(x)*mody)
            cos.append(res)
        else :
            cos.append(-1)

    _, position = find_topk(cos,topk)
    
    if cos[position[topk-1]] < mincos:
        k = topk - 2
        while k >= 0 and cos[position[k]] < mincos:
            k -= 1
        if k == -1:
            return -1, -1, y
        topk = k

    for i in range(topk):
        mask[position[i]] = 1

    vector_sum = deepcopy(X[position[0]])
    for i in range(1,topk):
        for j in range(len(vector_sum)):
            vector_sum[j] += X[position[i]][j]

    ans_index = -1
    ans_cos = -1
    vector_sum_len = np.linalg.norm(vector_sum)
    for i in range(topk):
        res = np.dot(X[position[i]],vector_sum)/(np.linalg.norm(X[position[i]])*vector_sum_len)
        if res > ans_cos :
            ans_cos = res
            ans_index = position[i]
    weight = vector_sum_len / np.linalg.norm(X[ans_index])
    w[ans_index] = weight
    y = y - weight * np.array(X[ans_index])
    return ans_cos, ans_index, y

def OO_scale_choose(arg,model_info):
    X_train = arg[1]
    dtrain = xgb.DMatrix(X_train)
    y_train = deepcopy(arg[2])
    y_train = y_train.astype('float64')
    y_train[y_train==0] = -1
    y_train[y_train==1] = 1

    base = arg[4]
    used = arg[5]
    retain = arg[3] - base

    if used == 0:
        used = model_info[1]
    
    # 用train data 和 model 得到xgb预测的输出结果
    traindata_output = []
    traindata_output.append([])
    for i in range(len(y_train)):
        traindata_output[0].append(0)
        

    tmp_ans = traindata_output[0]
    for i in range(1, used + 1):
        traindata_output.append([])
        ans = model_info[0].predict(dtrain, ntree_limit=i, output_margin=True)
        # for xxx in range(len(ans)) :
        #     ans[xxx] = np.log(ans[xxx]/(1-ans[xxx]))
        for j in range(len(ans)):
            traindata_output[i].append(ans[j] - tmp_ans[j])
        tmp_ans = ans

    # 用来算夹角的树
    traindata_for_OO = traindata_output[base+1:]

    # 在label中减去前base棵树的输出
    if base != 0:
        for i in range(len(y_train)):
            base_num = 0
            for j in range(1,base+1):
                base_num += traindata_output[j][i]
            y_train[i] -= base_num

    length = len(traindata_for_OO)
    mask = np.zeros(length,dtype=np.int).tolist()
    w = np.zeros(length,dtype=np.int).tolist()

    turn = 0
    for k in range(retain):
        cos, index, new_y = find_scale_cos(traindata_for_OO, y_train, mask, w, arg)
        y_train = new_y
        if index == -1:
            turn = k
            break
    for k in range(turn, retain):
        cos, index, new_y = find_max_cos(traindata_for_OO, y_train, w)
        y_train = new_y
    
    return w

def OO_scale_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    base = arg[4]
    used = arg[5]
    retain = arg[3] - base
    topk = arg[6]
    mincos = arg[7]

    w = OO_scale_choose(arg,model_info)

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(w)):
        if w[i] != 0:
            weight.append(w[i])
        else:
            weight.append(0)
    
    OOs = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            OOs += np.array(output[i]) * weight[i]

    acc = get_acc(OOs, y_test)
    mse = mean_squared_error(OOs, y_test)
    file.write("OOs_{}_{}_{} : mse={} ,acc={}\n".format(retain+base, used, base, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(retain+base,used))
    end_time = process_time()
    return OOs, {'name': 'OOs_{}_{}_{}_{}_{}'.format(retain+base, used, base, topk, mincos), \
        'acc': acc, 'mse': mse, 'chosen_num':retain + base, \
            'total_num':used,'time':end_time-start_time,\
                'weight':weight[1:]}

# ========================================================================
def OO_choose(arg,model_info):
    X_train = arg[1]
    dtrain = xgb.DMatrix(X_train)
    y_train = deepcopy(arg[2])
    y_train = y_train.astype('float64')
    y_train[y_train==0] = -1
    y_train[y_train==1] = 1

    base = arg[4]
    used = arg[5]
    retain = arg[3] - base

    if used == 0:
        used = model_info[1]
    
    # 用train data 和 model 得到xgb预测的输出结果
    traindata_output = []
    traindata_output.append([])
    for i in range(len(y_train)):
        traindata_output[0].append(0)

    tmp_ans = traindata_output[0]
    for i in range(1, used + 1):
        traindata_output.append([])
        ans = model_info[0].predict(dtrain, ntree_limit=i,output_margin=True)
        # for xxx in range(len(ans)) :
        #     ans[xxx] = np.log(ans[xxx]/(1-ans[xxx]))
        for j in range(len(ans)):
            traindata_output[i].append(ans[j] - tmp_ans[j])
        tmp_ans = ans

    # 用来算夹角的树
    traindata_for_OO = traindata_output[base+1:]

    cos = []
    mody = np.linalg.norm(y_train)
    for x in traindata_for_OO:
        res = np.dot(x,y_train)/(np.linalg.norm(x)* mody)
        cos.append(res)

    mask, _ = find_topk(cos,retain)

    return mask

def OO_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    base = arg[4]
    used = arg[5]
    retain = arg[3] - base

    mask = OO_choose(arg,model_info)

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i] != 0:
            weight.append(1)
        else:
            weight.append(0)
    
    OO = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            OO += np.array(output[i]) * weight[i]

    acc = get_acc(OO, y_test)
    mse = mean_squared_error(OO, y_test)
    file.write("OO_{}_{}_{} : mse={} ,acc={}\n".format(retain+base, used, base, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(retain+base,used))
    end_time = process_time()
    return OO, {'name': 'OO_{}_{}_{}'.format(retain+base, used, base), \
        'acc': acc, 'mse': mse, 'chosen_num':retain + base, \
            'total_num':used,'time':end_time-start_time,\
                'weight':weight[1:]}
# ========================================================================
def kappa(x1, x2):
    c = [[0 for i in range(2)] for j in range(2)]
    a = 0
    b = 0
    datasize = min(5000, len(x1))

    for i in range(datasize):
        if x1[i]>0 :
            a = 0
        else:
            a = 1
        if x2[i]>0 :
            b = 0
        else:
            b = 1
        c[a][b] += 1

    K1 = (c[0][0] + c[1][1])/2
    K2 = ( (c[0][0]+c[0][1])*(c[0][0]+c[1][0]) + (c[1][0]+c[1][1])*(c[0][1]+c[1][1]) ) / 4
    kappa_val = (K1 - K2) / (1 - K2)
    return kappa_val


def Kappa_choose(arg,model_info):
    y_train = deepcopy(arg[2])
    y_train = y_train.astype('float64')
    y_train[y_train==0] = -1
    y_train[y_train==1] = 1

    base = arg[4]
    used = arg[5]
    retain = arg[3] - base

    if used == 0:
        used = model_info[1]
    
    # 用train data 和 model 得到xgb预测的输出结果
    traindata_output = arg[6]

    traindata_for_kappa = traindata_output[base+1:used+1]
    length = len(traindata_for_kappa)
    kappa_sum = [0 for i in range(length)]

    for i in range(length):
        for j in range(i+1, length):
            temp = kappa(traindata_for_kappa[i], traindata_for_kappa[j])
            kappa_sum[i] += temp
            kappa_sum[j] += temp
            # print(i,j)
    
    mask, position = find_mink(kappa_sum, retain)
    return mask

def Kappa_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    base = arg[4]
    used = arg[5]
    retain = arg[3] - base

    mask = Kappa_choose(arg,model_info)

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i] != 0:
            weight.append(1)
        else:
            weight.append(0)
    
    kappa = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            kappa += np.array(output[i]) * weight[i]

    acc = get_acc(kappa, y_test)
    mse = mean_squared_error(kappa, y_test)
    file.write("kappa_{}_{}_{} : mse={} ,acc={}\n".format(retain+base, used, base, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(retain+base,used))
    end_time = process_time()
    return kappa, {'name': 'kappa_{}_{}_{}'.format(retain+base, used, base), \
        'acc': acc, 'mse': mse, 'chosen_num':retain + base, \
            'total_num':used,'time':end_time-start_time,\
                'weight':weight[1:]}



# ========================================================================
"""
    说明回调函数接口：
        参数output是对dtest的预测结果
        output[i][j]表示第i棵树，接收第j个样本的输出结果，注意output[0]里面全是0
        arg：计算可能需要的数值，需事先设定好
        file：传入一个文件指针，用来打印结果
        y_test：样本的真实标签
        返回值：第一个是预测的输出，第二个是统计参数
"""


# num=arg[1]表示计算前num棵树
def origin_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    num=arg[1]
    origin = np.array(output[0],dtype=np.float64)
    for i in range(1, num+1):
        origin += np.array(output[i])
    weight = np.ones(num,dtype=np.int).tolist()
    acc = get_acc(origin,y_test)
    mse=mean_squared_error(origin, y_test)
    file.write("origin_{} mse:{} ,acc={}\n".format(num, mse, acc))
    end_time = process_time()
    return origin, {'name':'origin_{}'.format(num),'acc':acc,'mse':mse,\
        'time':end_time-start_time,'weight':weight,'chosen_num':num, \
            'total_num':model_info[1],}

'''
    mean_tar=arg[1] 表示训练数据真实label的平均值
    num=arg[2] 表示最多取到第几棵树
    base = arg[3] 表示前几棵树需要全选
    例如 base=3 num=10 对于kplus_odd 来说，就是选 1，2，3，5，7，9
'''
def kplus_odd_cb(output,arg,file,y_test,model_info): #k+1法
    start_time = process_time()
    mean_tar=arg[1]
    used = arg[2]
    base=arg[3]

    weight = np.zeros(used+1,dtype=np.int).tolist()

    sum = np.array(output[1])
    mean=[0]
    # 计算平均输出
    mean.append(np.array(sum).mean())
    for i in range(len(output)-2):
        sum+=np.array(output[i+2])
        mean.append(np.array(sum).mean())

    for i in range(1,base+1):
        weight[i] = 1
    # 计算后续的估计部分， begin是第一个比base大的奇数
    begin = base + 1 + base % 2
    if begin==1:
        weight[1] = 1
        begin += 2
    for i in range(begin, used+1, 2):
        k = (mean_tar - mean[i-1]) / (mean_tar - mean[i])
        weight[i] = 1 + k

    calcu = np.array(output[0],dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            calcu += np.array(output[i]) * weight[i]

    acc = get_acc(calcu, y_test)
    mse = mean_squared_error(calcu, y_test)
    file.write("k+1_odd_{}_{} : mse={} ,acc={}\n".format(base, used, mse, acc))
    end_time = process_time()
    return calcu,{'name':'kplus_odd_{}_{}'.format(base,used),'acc':acc,\
        'mse':mse,'time':end_time-start_time,'weight':weight[1:],'chosen_num':(used - base)//2 + base, \
            'total_num':used,}

# 和上面一样
def kplus_even_cb(output,arg,file,y_test,model_info): #k+1法
    start_time = process_time()
    mean_tar=arg[1]
    used = arg[2]
    base=arg[3]

    weight = np.zeros(used+1,dtype=np.int).tolist()

    sum = np.array(output[1])
    mean=[0]
    # 计算平均输出
    mean.append(np.array(sum).mean())
    for i in range(len(output)-2):
        sum+=np.array(output[i+2])
        mean.append(np.array(sum).mean())

    for i in range(1,base+1):
        weight[i] = 1
    # 计算后续的估计部分， begin是第一个比base大的偶数
    begin = base + 2 - base % 2
    for i in range(begin, used+1, 2):
        k = (mean_tar - mean[i-1]) / (mean_tar - mean[i])
        weight[i] = 1 + k

    calcu = np.array(output[0],dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            calcu += np.array(output[i]) * weight[i]

    acc = get_acc(calcu, y_test)
    mse = mean_squared_error(calcu, y_test)
    file.write("k+1_even_{}_{} : mse={} ,acc={}\n".format(base, used, mse, acc))
    end_time = process_time()
    return calcu,{'name':'kplus_even_{}_{}'.format(base,used),'acc':acc,\
        'mse':mse,'time':end_time-start_time,'weight':weight[1:],'chosen_num':(used - base + 1)//2 + base, \
            'total_num':used,}


'''
    n_round=arg[1] 表示总训练轮次
    base=arg[2] 表示必定取前base个
'''
def odd_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    used = arg[1]
    base = arg[2]

    weight = np.zeros(used+1,dtype=np.int).tolist()

    for i in range(1, base+1):
        weight[i] = 1

    # 计算后续的奇数部分
    begin = base + 1 + base % 2
    for i in range(begin, used+1, 2):
        weight[i] = 1

    odd = np.array(output[0],dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            odd += np.array(output[i]) * weight[i]

    acc = get_acc(odd, y_test)
    mse = mean_squared_error(odd, y_test)
    file.write("odd_{}_{} : mse={} ,acc={}\n".format(base, used, mse, acc))
    end_time = process_time()
    return odd,{'name':'odd_{}_{}'.format(base,used),'acc':acc,\
        'mse':mse,'time':end_time-start_time,'weight':weight[1:],'chosen_num':(used - base)//2 + base, \
            'total_num':used,}


def even_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    used = arg[1]
    base = arg[2]

    weight = np.zeros(used+1,dtype=np.int).tolist()

    for i in range(1, base+1):
        weight[i] = 1

    # 计算后续的偶数部分
    begin = base + 2 - base % 2
    for i in range(begin, used+1, 2):
        weight[i] = 1

    even = np.array(output[0],dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            even += np.array(output[i]) * weight[i]

    acc = get_acc(even, y_test)
    mse = mean_squared_error(even, y_test)
    file.write("even_{}_{} : mse={} ,acc={}\n".format(base, used, mse, acc))
    end_time = process_time()
    return even,{'name':'even_{}_{}'.format(base,used),'acc':acc,\
        'mse':mse,'time':end_time-start_time,'weight':weight[1:],'chosen_num':(used - base + 1)//2 + base, \
            'total_num':used,}


def _lasso_chose_by_threshold(arg,model_info):
    X_train = arg[1]
    dtrain = xgb.DMatrix(X_train)
    y_train = deepcopy(arg[2])
    y_train = y_train.astype('float')
    y_train[y_train==0] = -1
    y_train[y_train==1] = 1
    threshold = arg[3]
    base = arg[4]
    used = arg[5]
    if used == 0:
        used = model_info[1]

    # 用train data 和 model 得到xgb预测的输出结果
    traindata_output = []
    traindata_output.append([])
    for i in range(len(y_train)):
        traindata_output[0].append(0)

    tmp_ans = traindata_output[0]
    for i in range(1, used + 1):
        traindata_output.append([])
        ans = model_info[0].predict(dtrain, ntree_limit=i)
        for xxx in range(len(ans)) :
            ans[xxx] = np.log(ans[xxx]/(1-ans[xxx]))
        for j in range(len(ans)):
            traindata_output[i].append(ans[j] - tmp_ans[j])
        tmp_ans = ans

    # 处理一下traindata_output（相当于每个树的输出）
    # 舍去前base棵树，然后转置
    traindata_for_lasso = traindata_output[base+1:]
    traindata_for_lasso = np.transpose(traindata_for_lasso)

    # 在label中减去前base棵树的输出
    for i in range(len(y_train)):
        base_num = 0
        for j in range(1,base+1):
            base_num += traindata_output[j][i]
        y_train[i] -= base_num

    # 搜索lasso最佳参数
    alpha_range = np.logspace(-10, -2, 10, base=10)
    lasso_model = LassoCV(alphas=alpha_range,cv=5).fit(traindata_for_lasso,y_train)
    
    best_alpha = lasso_model.alpha_
    w = lasso_model.coef_
    wm = np.maximum(w, -w)
    avg = np.average(wm)

    mask =[]
    position = []
    for idx,val in enumerate(wm):
        if  val >= threshold * avg:
            mask.append(1)
            position.append(idx)
        else :
            mask.append(0)
    
    return mask, position, w, traindata_for_lasso


def _lasso_chose_by_sort(arg,model_info):
    X_train = arg[1]
    dtrain = xgb.DMatrix(X_train)
    y_train = deepcopy(arg[2])
    y_train = y_train.astype('float64')
    y_train[y_train==0] = -1
    y_train[y_train==1] = 1

    base = arg[4]
    used = arg[5]
    if used == 0:
        used = model_info[1]
    
    retain = arg[3]
    if retain == 0:
        if used > 500:
            retain = math.ceil(used*0.3)
        elif used > 200:
            retain = max(60,math.ceil(used*0.2))
        elif used > 50:
            retain = max(15,math.ceil(used*0.2))
        else:
            retain = max(5,math.ceil(used*0.3))
    if retain <= base:
        retain = base + 1

    # 用train data 和 model 得到xgb预测的输出结果
    traindata_output = []
    traindata_output.append([])
    for i in range(len(y_train)):
        traindata_output[0].append(0)

    tmp_ans = traindata_output[0]
    for i in range(1, used + 1):
        traindata_output.append([])
        ans = model_info[0].predict(dtrain, ntree_limit=i)
        for xxx in range(len(ans)) :
            ans[xxx] = np.log(ans[xxx]/(1-ans[xxx]))
        for j in range(len(ans)):
            traindata_output[i].append(ans[j] - tmp_ans[j])
        tmp_ans = ans

    # 处理一下traindata_output（相当于每个树的输出）
    # 舍去前base棵树，然后转置
    traindata_for_lasso = traindata_output[base+1:]
    traindata_for_lasso = np.transpose(traindata_for_lasso)

    # 在label中减去前base棵树的输出
    for i in range(len(y_train)):
        base_num = 0
        for j in range(1,base+1):
            base_num += traindata_output[j][i]
        y_train[i] -= base_num

    # 搜索lasso最佳参数
    alpha_range = np.logspace(-10, -2, 10, base=10)
    traindata_for_lasso = traindata_for_lasso.astype('float64')
    lasso_model = LassoCV(alphas=alpha_range,cv=5).fit(traindata_for_lasso,y_train)

    best_alpha = lasso_model.alpha_
    w = lasso_model.coef_
    wm = np.maximum(w, -w)

    mask, position = find_topk(wm, retain - base)

    return retain, mask, position, w, traindata_for_lasso


def lasso_p_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    threshold = arg[3]
    base = arg[4]
    used = arg[5]
    if used == 0:
        used = model_info[1]
    mask, position, w, _ = _lasso_chose_by_threshold(arg,model_info)

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i]==1:
            weight.append(1)
        else:
            weight.append(0)

    lasso = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            lasso += np.array(output[i]) * weight[i]

    acc = get_acc(lasso, y_test)
    mse = mean_squared_error(lasso, y_test)
    file.write("lasso_p_{}_{}_{} : mse={} ,acc={}\n".format(base, threshold, used, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(np.sum(mask)+base,used))
    end_time = process_time()
    return lasso, {'name': 'lasso_p_{}_{}_{}'.format(base, threshold, used), \
        'acc': acc, 'mse': mse, 'chosen_num':np.sum(mask)+base, \
            'total_num':used,'time':end_time-start_time,\
                'weight':weight[1:]}



def lasso_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    threshold = arg[3]
    base = arg[4]
    used = arg[5]
    if used == 0:
        used = model_info[1]
    mask, position, w, _ = _lasso_chose_by_threshold(arg,model_info)

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i]==1:
            weight.append(w[i])
        else:
            weight.append(0)

    lasso = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            lasso += np.array(output[i]) * weight[i]

    acc = get_acc(lasso, y_test)
    mse = mean_squared_error(lasso, y_test)
    file.write("lasso_{}_{}_{} : mse={} ,acc={}\n".format(base, threshold, used, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(np.sum(mask)+base,used))
    end_time = process_time()
    return lasso, {'name': 'lasso_{}_{}_{}'.format(base, threshold, used), \
        'acc': acc, 'mse': mse, 'chosen_num':np.sum(mask)+base, \
            'total_num':used,'time':end_time-start_time,\
                'weight':weight[1:]}



def lasso_linear_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    threshold = arg[3]
    y_train = arg[2]
    y_train=y_train.astype('float')
    base = arg[4]
    used = arg[5]
    if used == 0:
        used = model_info[1]
    mask, position, w, traindata_for_lasso = _lasso_chose_by_threshold(arg,model_info)

    traindata_for_linear = []
    for line in traindata_for_lasso:
        temp = []
        for i in range(len(position)):
            temp.append(line[position[i]])
        traindata_for_linear.append(temp)

    linear = LinearRegression().fit(traindata_for_linear,y_train)

    wl = linear.coef_

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i]==1:
            weight.append(wl[ position.index(i) ])
        else:
            weight.append(0)

    lasso = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            lasso += np.array(output[i]) * weight[i]

    acc = get_acc(lasso, y_test)
    mse = mean_squared_error(lasso, y_test)
    file.write("lasso_linear_{}_{}_{} : mse={} ,acc={}\n".format(base, threshold, used, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(np.sum(mask)+base,used))
    end_time = process_time()
    return lasso, {'name': 'lasso_L_{}_{}_{}'.format(base, threshold, used), \
        'acc': acc, 'mse': mse, 'chosen_num':np.sum(mask)+base, \
            'total_num':used,'time':end_time-start_time,\
                'weight':weight[1:]}




def lasso_sort_p_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    base = arg[4]
    used = arg[5]
    if used == 0:
        used = model_info[1]
    
    retain ,mask, position, w, _ = _lasso_chose_by_sort(arg,model_info)

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i]==1:
            weight.append(1)
        else:
            weight.append(0)

    lasso = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            lasso += np.array(output[i]) * weight[i]

    acc = get_acc(lasso, y_test)
    mse = mean_squared_error(lasso, y_test)
    file.write("lasso_sort_p_{}_{}_{} : mse={} ,acc={}\n".format(base, retain, used, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(np.sum(mask)+base,used))
    end_time = process_time()
    return lasso, {'name': 'lasso_sort_p_{}_{}_{}'.format(base, retain, used), \
        'acc': acc, 'mse': mse, 'chosen_num':np.sum(mask)+base, \
            'total_num':used,'time':end_time-start_time,\
                'weight':weight[1:]}



def lasso_sort_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    base = arg[4]
    used = arg[5]
    if used == 0:
        used = model_info[1]

    retain ,mask, position, w, _ = _lasso_chose_by_sort(arg,model_info)

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i]==1:
            weight.append(w[i])
        else:
            weight.append(0)

    lasso = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            lasso += np.array(output[i]) * weight[i]

    acc = get_acc(lasso, y_test)
    mse = mean_squared_error(lasso, y_test)
    file.write("lasso_sort_{}_{}_{} : mse={} ,acc={}\n".format(base, retain, used, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(np.sum(mask)+base,used))
    end_time = process_time()
    return lasso, {'name': 'lasso_sort_{}_{}_{}'.format(base, retain, used), \
        'acc': acc, 'mse': mse, 'chosen_num':np.sum(mask)+base, \
            'total_num':used,'time':end_time-start_time,\
                'weight':weight[1:]}



def lasso_sort_linear_cb(output,arg,file,y_test,model_info):
    start_time = process_time()
    y_train = arg[2]
    y_train=y_train.astype('float')
    base = arg[4]
    used = arg[5]
    if used == 0:
        used = model_info[1]
    
    retain ,mask, position, w, traindata_for_lasso = _lasso_chose_by_sort(arg,model_info)

    traindata_for_linear = []
    for line in traindata_for_lasso:
        temp = []
        for i in range(len(position)):
            temp.append(line[position[i]])
        traindata_for_linear.append(temp)

    linear = LinearRegression().fit(traindata_for_linear,y_train)

    wl = linear.coef_

    weight = [0]
    for i in range(base):
        weight.append(1)
    for i in range(len(mask)):
        if mask[i]==1:
            weight.append(wl[ position.index(i) ])
        else:
            weight.append(0)

    lasso = np.array(output[0], dtype=np.float64)
    for i in range(len(weight)):
        if weight[i] != 0:
            lasso += np.array(output[i]) * weight[i]

    acc = get_acc(lasso, y_test)
    mse = mean_squared_error(lasso, y_test)
    file.write("lasso_sort_linear_{}_{}_{} : mse={} ,acc={}\n".format(base, retain, used, mse, acc))
    file.write("\t{} is chosen from {} trees\n".format(np.sum(mask)+base,used))
    end_time = process_time()
    return lasso, {'name': 'lasso_sort_L_{}_{}_{}'.format(base, retain, used), \
        'acc': acc, 'mse': mse, 'chosen_num':np.sum(mask)+base, \
            'total_num':used,'time':end_time-start_time,\
                'weight':weight[1:]}











# pred是一棵树对于所有数据输出的平均值
'''
    draw_cb 的参数设置
    arg=[[ ["draw",0], 树的棵数, 步长 ]]
    arg_param=[]
'''
def drawdata_cb(output,arg,file,y_test,model_info):
    tree_num = arg[1]
    step = arg[2]
    residual = []
    acc = []
    y_mean = np.mean(y_test)
    memo = np.zeros(len(y_test),dtype=np.float)
    for i in range(1,tree_num+1,1):
        tree_out = np.array(output[i])
        if i==1:
            cur_pred = memo + tree_out
        else:
            cur_pred = memo + tree_out * 10
        memo += tree_out
        cur_acc = get_acc(cur_pred,y_test)
        cur_residual = get_residual(cur_pred,y_test)
        acc.append(cur_acc)
        residual.append(cur_residual)

    return None,{'name':'draw','residual_draw':residual,'acc_draw':acc, 'step':step, 'tree_num':tree_num, 'outfile':arg[3]}



###################################################
# statistic_cb
###################################################

import os
def time_cb(rets,path,info):
    flag = 0    
    if os.path.exists(info['output_path'] + "time.csv")==False:
        flag = 1

    f=open(path+"/summary.txt","a")
    f.write("report : time\n")

    table=open(info['output_path'] + "time.csv","a")
    if(flag == 1):
        for i in range(len(rets[0])):
            if 'time' in rets[0][i].keys():
                table.write(",{}".format(rets[0][i]['name']))
        table.write("\n")
    table.write('{}'.format(info['output_dic']))


    time=[]
    name=[]

    for i in range(len(rets[0])):    # 有几个回调
        if 'time' in rets[0][i].keys():
            temp=[]
            name.append(rets[0][i]['name'])
            for j in range(len(rets)):      # 对于该回调的10fold
                temp.append(rets[j][i]['time'])
            time.append(np.array(temp).mean())

    for i in range(len(time)):
        f.write("{}: \n\ttime cost = {}\n".format(name[i],time[i]))
        table.write(",{:.5f}".format(time[i]))


    f.write('\n')
    f.close()
    table.write("\n")
    table.close()

def weight_cb(rets,path,info):
    table=open(info['output_path'] + "weight.csv","a")

    dic_name = info['output_dic']
    for i in range(len(rets[0])):    # 有几个回调
        name = rets[0][i]['name']
        for j in range(len(rets)):   # 有几个fold
            if 'weight' in rets[j][i].keys():
                weight = rets[j][i]['weight']
                table.write('{},{},fold{}'.format(dic_name,name,j))
                for val in weight:
                    table.write(',{}'.format(val)) 
                table.write('\n')
    table.close()



def chosen_num_cb(rets,path,info):
    flag = 0    
    if os.path.exists(info['output_path'] + "tree_num.csv")==False:
        flag = 1
    f = open(path + "/summary.txt", "a")
    f.write("report : chosen number of trees : \n")

    table=open(info['output_path'] + "tree_num.csv","a")
    if(flag == 1):
        for i in range(len(rets[0])):
            if 'chosen_num' in rets[0][i].keys():
                table.write(",{}".format(rets[0][i]['name']))
        table.write("\n")
    table.write('{}'.format(info['output_dic']))

    chosen_num = []
    total_num = []
    name = []

    for i in range(len(rets[0])):
        if 'chosen_num' in rets[0][i].keys():
            temp=[]
            name.append(rets[0][i]['name'])
            for j in range(len(rets)):
                temp.append(rets[j][i]['chosen_num'])
            chosen_num.append(np.array(temp).mean())

        if 'total_num' in rets[0][i].keys():
            total_num.append(rets[0][i]['total_num'])

    for i in range(len(chosen_num)):
        f.write("{}: \n\t{} in total {} trees\n".format(name[i],chosen_num[i],total_num[i]))
        table.write(",{:.3f}".format(chosen_num[i]))

    f.write('\n')
    f.close()
    table.write("\n")
    table.close()

def acc_cb(rets,path,info):
    flag = 0    
    if os.path.exists(info['output_path'] + "acc.csv")==False:
        flag = 1
    f=open(path+"/summary.txt","a")
    f.write("report : avg acc\n")

    table=open(info['output_path'] + "acc.csv","a")
    table_std=open(info['output_path'] + "acc_std.csv","a")
    
    if(flag == 1):
        for i in range(len(rets[0])):
            if 'acc' in rets[0][i].keys():
                table.write(",{}".format(rets[0][i]['name']))
                table_std.write(",{}".format(rets[0][i]['name']))
        table.write("\n")
        table_std.write("\n")
    
    table.write('{}'.format(info['output_dic']))
    table_std.write('{}'.format(info['output_dic']))

    acc=[]
    std=[]
    name=[]

    for i in range(len(rets[0])):    # 有几个回调
        if 'acc' in rets[0][i].keys():
            temp=[]
            name.append(rets[0][i]['name'])
            for j in range(len(rets)):      # 对于该回调的10fold
                temp.append(rets[j][i]['acc'])
            acc.append(np.array(temp).mean())
            std.append(np.std(temp))

    for i in range(len(acc)):
        f.write("{}: {}\n".format(name[i],acc[i]))
        table.write(",{:.3f}".format(acc[i]*100))
        table_std.write(",{:.3f}".format(std[i]*100))

    f.write('\n')
    f.close()
    table.write("\n")
    table.close()
    table_std.write("\n")
    table_std.close()



def draw_cb(rets,path,info):

    acc=[]
    residual=[]
    outfile=[]

    tree_num = 0
    step = 0

    for i in range(len(rets[0])):
        if 'acc_draw' in rets[0][i].keys():
            temp_acc = []
            temp_residual = []
            outfile.append(rets[0][i]['outfile'])
            tree_num = rets[0][i]['tree_num']
            step = rets[0][i]['step']

            length = len(rets[0][i]['acc_draw'])
            for k in range(length):
                for j in range(len(rets)):      # 对于该回调的10fold
                    temp_acc.append(rets[j][i]['acc_draw'][k])
                    temp_residual.append(rets[j][i]['residual_draw'][k])

                acc.append(np.array(temp_acc).mean())
                residual.append(np.array(temp_residual).mean())

    from matplotlib.ticker import FormatStrFormatter 
    fontsize = 28
    fig,ax1 = plt.subplots()
    fig.set_size_inches(7.5,5)
    plt.xlabel('Number of trees',fontsize=fontsize)

    acc_max = np.max(acc)
    acc_min = np.min(acc)

    # ax1.plot(range(0,tree_num,step),acc[::step],'.-',label='Accuracy',color='dodgerblue',linewidth=0.8)
    ax1.plot(range(0,251,10),acc[0:251:10],'.-',label='Accuracy',color='dodgerblue',linewidth=0.8)
    ax1.set_ylabel('Accuracy',fontsize=fontsize)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.set_ylim((acc_min-0.1,acc_max+0.03))

    ax2=ax1.twinx()
    # ax2.plot(range(0,tree_num,step),residual[::step],'x-',label='Residual',color='red',linewidth=0.8)
    ax2.plot(range(0,251,10),residual[0:251:10],'x-',label='Residual',color='red',linewidth=0.8)
    ax2.set_ylabel('Residual',fontsize=fontsize)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1+handles2, labels1+labels2,loc='center right',fontsize=26)

    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontsize(20) for label in labels]

    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontsize(20) for label in labels]

    title = info['dataset'].title()
    plt.title(title,fontsize = fontsize)
    plt.savefig(info['output_path']+outfile[0]+'.pdf',dpi = 400,bbox_inches='tight')
