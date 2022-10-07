import sys
import pickle
import os
dataset = sys.argv[1]
output = sys.argv[2]
tree_num = sys.argv[3]
depth = sys.argv[4]
tree_dic = './dump_tree/'
if os.path.exists(tree_dic) == False:
    os.mkdir(os.getcwd() + "/" + tree_dic)
model_path = '../xgboost_models/{}_{}_{}/'.format(dataset,tree_num,depth)
for i in range(10):
    model_name = '{}_fold{}.dat'.format(dataset,i)
    model = pickle.load(open(model_path + model_name, "rb"))
    output_name = '{}_fold{}.txt'.format(output,i)
    model.dump_model(tree_dic + output_name)
