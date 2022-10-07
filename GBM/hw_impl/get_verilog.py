import sys,os
sys.path.append('./utils/')
from Tree_Reader import tree_reader,add_weights,load_weights
from Build_vlg import Build_adder_without_cin,Build_pipeline,Build_top,Build_xdc,Builder
from LGB_Tree_Reader import lgb_tree_reader
if __name__ == '__main__':

    # hardware_path = './hardware/'
    hardware_path = './hardware_xgb/'
    if len(sys.argv)!=6:
         hardware_path = './hardware_lgb/'
    tree_path = './dump_tree/'
    # cp ../xgboost_/xgboost_output_base/weight.csv data/weight.csv
    weights_path = './data'
    if os.path.exists(weights_path) == False:
        os.mkdir(os.getcwd() + "/" + weights_path)
   
    

    # cd hardware_path
    # ls | sed 's@^@@g' | paste -s -d " "| xargs -n 1 cp ../../run.tcl
    # nohup vivado -mode batch -source run.tcl &
    dataset = sys.argv[1]
    fold = sys.argv[2]
    reduction = sys.argv[3]
    input_num = int(sys.argv[4])
    output = hardware_path + sys.argv[5]
    
    if os.path.exists(hardware_path) == False:
        os.mkdir(hardware_path)
    if os.path.exists(output) == False:
        os.mkdir(output)

    param = {
        'IN_IW' : 2,    # 整数部分位宽
        'IN_DW' : 1,    # 小数部分位宽
        'IN_SIGN': 0,   # 是否需要符号位
        'OUT_IW': 3,
        'OUT_DW': 12,
        'OUT_SIGN':1,
        'INPUT_NUM':input_num
    }

    if len(sys.argv)==6:
        weight_path = './data/weight_xgb.csv'
        os.system('cp ../xgboost_output/weight.csv ' +weight_path)
        tree_file = tree_path + dataset + '_' + fold +'.txt'
        trees = tree_reader(tree_file)
    # for lgb
    else:
        # for lgb
        weight_path = './data/weight_lgb.csv'
        os.system('cp ../lgb_output/weight.csv ' +weight_path)
        model_path = '../lgb_models/{}_{}/'.format(dataset[:len(dataset)-1], 200)
        model_file = "{}{}_{}.dat".format(model_path, dataset[:len(dataset)-1], fold)
        trees = lgb_tree_reader(model_file)
        # pass

    weights = load_weights(weight_path, dataset, reduction, fold)
   
    trees = add_weights(trees, weights[:len(trees)])

    tree_num = len(trees)

    doc_path= output + '/trees'
    Builder(trees, doc_path, param, dataset)

    tree_input_width = param['IN_IW'] + param['IN_DW'] + param['IN_SIGN']
    tree_output_width = param['OUT_IW'] + param['OUT_DW'] + param['OUT_SIGN']

    Build_adder_without_cin(width = tree_output_width ,file_path=  output + '/adder.v')
    Build_pipeline(input_width = tree_output_width, input_num=tree_num, file_path=output+'/pipeline.v')
    Build_top(input_width=tree_input_width,input_num=input_num,output_width=tree_output_width,\
        tree_num=tree_num,file_path=output+'/top.v')
    Build_xdc(inputnum=input_num,inputwidth=tree_input_width,outputwidth=tree_output_width,\
        pin_file='./data/pin_name', file_path=output + '/top.xdc')