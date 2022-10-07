def bin_trees(trees,k):
    leaflist = get_leaf(trees)
    sort_leaf(leaflist,0,len(leaflist))

    maxw = leaflist[0].w
    minw = leaflist[-1].w

    half = k//2
    pos_unit = maxw / half
    neg_unit = minw / half

    # if k != 4:
    pos_list = []
    neg_list = []
    for i in range(half):
        pos_list.append((half - i) * pos_unit)
        neg_list.append((half - i) * neg_unit)
    pos_list.append(0)
    neg_list.append(0)
    
    for i in range(len(leaflist)):
        if leaflist[i].w > 0:
            for j in range(len(pos_list) - 1):
                if leaflist[i].w > pos_list[j + 1]:
                    leaflist[i].w = pos_list[j]
                    break
        elif leaflist[i].w < 0:
            for j in range(len(neg_list) - 1):
                if leaflist[i].w < neg_list[j + 1]:
                    leaflist[i].w = neg_list[j]
                    break
    # else:
    #     for i in range(len(leaflist)):
    #         if leaflist[i].w > maxw/2:
    #             leaflist[i].w = maxw
    #         elif leaflist[i].w > 0:
    #             leaflist[i].w = maxw/2
    #         elif leaflist[i].w == 0:
    #             leaflist[i].w = 0
    #         elif leaflist[i].w < minw/2:
    #             leaflist[i].w = minw
    #         elif leaflist[i].w < 0:
    #             leaflist[i].w = minw/2

def avg_trees(trees):
    leaflist = get_leaf(trees)
    sort_leaf(leaflist,0,len(leaflist))

    leaf_num = len(leaflist)
    group_size = 5
    group_num = leaf_num // group_size
    idx = 0

    for g in range(group_num):
        summ = 0
        for i in range(group_size):
            summ += leaflist[idx].w
            idx += 1
        avg = summ / group_size
        for i in range(group_size):
            leaflist[idx - 1 - i].w = avg

def get_leaf(trees):
    tree_num = len(trees)
    leaflist = []
    for i in range(tree_num):
        get_stats(trees[i],leaflist)
    return leaflist

def get_stats(node,leaflist):
    if node.leaf==True:
        leaflist.append(node)
    else:
        get_stats(node.left,leaflist)
        get_stats(node.right,leaflist)

def sort_leaf(stats,left,right):
    if right > left:
        bound = left
        p = left + 1
        while p < right : 
            if cmp(stats, p, left):
                bound += 1
                swap(stats, bound, p)
            p += 1
        swap(stats, left, bound)
        sort_leaf(stats, left, bound)
        sort_leaf(stats, bound+1, right)

def swap(xxx, a, b):
    temp = xxx[a]
    xxx[a] = xxx[b]
    xxx[b] = temp

def cmp(xxx, a, b):
    if xxx[a].w > xxx[b].w :
        return True
    else :
        return False