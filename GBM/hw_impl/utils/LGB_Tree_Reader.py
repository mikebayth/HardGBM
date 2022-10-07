# -*- coding: utf-8 -*-
# @Author  : LongyunBian
# @Time    : 2022/5/25 16:19
import sys

# sys.path.append('./utils/')
from Tree_Reader import Node
import pickle

# dataset = sys.argv[1]
# output = sys.argv[2]
# tree_num = sys.argv[3]
# depth = sys.argv[4]
# tree_dic = './dump_tree/'
#
# model_path = '../xgboost_/lgb_models/{}_{}/'.format(dataset, tree_num)


def _read_tree(tree_dict, node):

    if len(tree_dict.keys())==12:
        left_id_key = [i for i in tree_dict['left_child'].keys() if 'index' in i][0]
        right_id_key = [i for i in tree_dict['right_child'].keys() if 'index' in i][0]
        # print(left_id_key, right_id_key)
        node.set_node(tree_dict['split_feature'], tree_dict['threshold'], tree_dict['left_child'][left_id_key],
                      tree_dict['right_child'][right_id_key], missing=0)
        _read_tree(tree_dict['left_child'], node.left)
        _read_tree(tree_dict['right_child'], node.right)
    else:
        node.set_leaf(tree_dict['leaf_value'])
    # return tree


def lgb_tree_reader(model_file):
    model = pickle.load(open(model_file, "rb"))
    trees_dict = model.dump_model()
    trees = trees_dict['tree_info']
    nodes = []
    for tree in trees:
        tree = tree['tree_structure']
        node = Node(0, None)
        _read_tree(tree_dict=tree, node=node)
        nodes.append(node)
    return nodes
#
if __name__ == '__main__':
    # '../xgboost_/lgb_models/mobile_200/mobile_folo0.dat'
    lgb_tree_reader('../xgboost_/lgb_models/mobile_200/mobile_fold0.dat')