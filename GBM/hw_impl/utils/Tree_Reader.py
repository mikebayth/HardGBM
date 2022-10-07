# -*- coding: utf-8 -*-
import re
import os
import numpy as np

class Node:
    def __init__(self,id,father):
        self.id=id
        self.father=father
        self.leaf=None
        # for leaf
        self.w=None
        # below for node
        self.f_id=None
        self.val=None
        self.right=None
        self.left=None
        self.missing=None

    def set_leaf(self,w):
        self.leaf=True
        self.w=w

    def set_node(self,f_id,val,left_id,right_id,missing):
        self.leaf=False
        self.left=Node(left_id,self)
        self.right=Node(right_id,self)
        self.val=val
        if missing == left_id:
            self.missing = 0
        else:
            self.missing = 1
        self.f_id=f_id

def _read_leaf(line):
    str = re.findall(r'(.+?):leaf=(.+)', line)
    str=str[0]
    node_id = int(str[0])
    w = float(str[1])
    return node_id,w

def _read_node(line):
    ans = re.findall(r'(.+?):\[f(.+?)<(.+?)\] yes=(.+?),no=(.+?),missing=(.+?)', line)
    str=ans[0]
    node_id=int(str[0])
    f_id=int(str[1])
    val=float(str[2])
    left_id=int(str[3])
    right_id=int(str[4])
    missing=int(str[5])
    return node_id,f_id,val,left_id,right_id,missing

def _load_tree(lines,i):
    stack=[]
    tree=Node(0,None)

    if lines[i].find('leaf') >= 0:
        node_id, w = _read_leaf(lines[i])
        tree.set_leaf(w)
    else:
        node_id, f_id, val, left_id, right_id, missing \
            = _read_node(lines[i])
        tree.set_node(f_id, val, left_id, right_id, missing)
        stack.append(tree.right)
        stack.append(tree.left)

    while len(stack)!=0:
        i += 1
        cur_node=stack.pop()
        if lines[i].find('leaf')>=0:
            node_id,w=_read_leaf(lines[i])
            cur_node.set_leaf(w)
        else:
            node_id, f_id, val, left_id, right_id, missing\
                =_read_node(lines[i])
            cur_node.set_node(f_id, val, left_id, right_id, missing)
            stack.append(cur_node.right)
            stack.append(cur_node.left)
    return tree,i+1

def tree_reader(file_name):
    with open(file_name, 'r',encoding='utf-8') as f:
        lines = f.readlines()
    trees=[]
    i=0
    while i < len(lines):
        while lines[i].startswith('booster')==False :# a new tree
            i+=1
        i+=1 #从booster下一行开始
        tree,i=_load_tree(lines,i)
        trees.append(tree)
    return trees


def print_tree(tree,file,depth):
    global count
    if tree!=None:
        if tree.leaf:
            for i in range(depth):
                file.write('\t')
            file.write('{} leaf = {} \n'.format(tree.id,tree.w))
        else:
            for i in range(depth):
                file.write('\t')
            file.write('{} [f{} < {}] \n'.format(tree.id,tree.f_id,tree.val))
            print_tree(tree.left, file, depth+1)
            print_tree(tree.right, file, depth + 1)


#######################################################################################

def report_node_num(root):
    if root == None:
        return 0,0,0
    elif root.leaf :
        return 1,0,1
    else :
        l_t, l_b, l_l = report_node_num(root.left)
        r_t, r_b, r_l = report_node_num(root.right)
        return l_t + r_t + 1, l_b + r_b + 1, l_l + r_l


def report_depth(root,depth,deeplist):
    if root == None:
        return
    elif root.leaf:
        deeplist.append(depth)
        return
    else:
        report_depth(root.left,depth+1,deeplist)
        report_depth(root.right,depth+1,deeplist)
        return

def _foo(num_leaf, deeplist):
    count_and = 0
    count_or = 0

    ans = 2
    bits = 1
    while (ans < num_leaf):
        ans = ans * 2
        bits += 1

    for i in range(bits):
        scope=pow(2,i)
        base=scope
        while base < num_leaf:
            if base+scope >= num_leaf:
                tmp=num_leaf - base
            else:
                tmp=scope
            for offset in range(tmp):
                count_and += deeplist[base+offset]
                count_or += 1
            base+=2*scope
        count_or -= 1
    return count_and,count_or



def report_tree_info(trees,file_path):
    file = open(file_path,'w')

    for id in range(len(trees)):
        tree = trees[id]
        total, branch, leaf = report_node_num(tree)
        deeplist = []
        report_depth(tree,0,deeplist)
        np.array(deeplist)
        for i in range(len(deeplist)):
            if deeplist[i] > 0:
                deeplist[i] -= 1

        file.write('tree{}:\n'.format(id))
        file.write('\ttotal node = {}, branch node = {}, leaf node = {}\n'.format(total,branch,leaf))

        file.write('\tonehot: ULB = {}, && = {}, || = {}\n'.format(2*branch,np.sum(deeplist),0))

        n1,n2 = _foo(leaf,deeplist)
        file.write('\tencoding: ULB = {}, && = {}, || = {}\n'.format(2 * branch - 1,n1,n2))
        file.write('\n')

    file.close()
    return
#######################################################################

def load_weights(weight_path,dataset,reduction,fold):
    with open(weight_path,'r') as f:
        lines = f.readlines()
    for line in lines :
        arr = line.split(',')
        if arr[0]==dataset and arr[1]==reduction and arr[2]==fold:
            return arr[3:]


def add_weight(tree,weight):
    if tree.leaf == True:
        tree.w = tree.w * weight
    else:
        add_weight(tree.left,weight)
        add_weight(tree.right,weight)

def add_weights(trees,weights):
    ret = []
    for i in range(len(weights)):
        weights[i] = float(weights[i])
        if weights[i]!=0:
            add_weight(trees[i],weights[i])
            ret.append(trees[i])
    return ret