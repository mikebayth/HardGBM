from Tree_Reader import tree_reader,report_tree_info
import sys,re,os
from progress.bar import Bar


def get_binary(num,width,sign):
    is_negetive = False
    if num < 0:
        val = -num
        is_negetive = True
    else:
        val = num
    int_val = int(val)
    str_val = bin(int_val)
    str_val = str_val[2:]
    while len(str_val)<width:
        str_val = '0' + str_val
    str_val = str_val[-width:]

    if sign ==1 and is_negetive:
        for c in str_val:
            pass

    if sign == 1:
        if is_negetive:
            str_val = '1' + str_val
        else:
            str_val = '0' + str_val

    str_val = '{}\'b'.format(width + sign) + str_val
    return str_val

def reversal(node,str_set,w_set,str_now,params):
    in_iw = params['IN_IW']
    in_dw = params['IN_DW']
    in_sign = params['IN_SIGN']
    if node.leaf:
        if len(str_now)==0:
            w_set.append(node.w)
            str_set.append('')
            return
        str_tmp="("+str_now[0]
        i=1
        while i< len(str_now):
            str_tmp=str_tmp+" && "+str_now[i]
            i+=1
        str_tmp = str_tmp+")"
        str_set.append(str_tmp)
        w_set.append(node.w)
    else:
        num = node.val
        for i in range(in_dw):
            num *= 2
        str="(f{} < {})".format(node.f_id,get_binary(num,in_iw+in_dw,in_sign))
        str_now.append(str)
        reversal(node.left,str_set,w_set,str_now,params)
        str_now.pop()
        str="(f{} >= {})".format(node.f_id,get_binary(num,in_iw+in_dw,in_sign))
        str_now.append(str)
        reversal(node.right,str_set,w_set,str_now,params)
        str_now.pop()

def get_str(tree,params):
    str_set=[]
    str_now=[]
    w_set=[]
    reversal(tree,str_set,w_set,str_now,params)
    return str_set,w_set

def get_bits(num):
    ans=2
    bits=1
    while(ans<num):
        ans=ans*2
        bits+=1
    return bits,ans

def get_onehot(num,length):
    tempstr = ""
    for i in range(length):
        if i == num:
            tempstr += '1'
        else:
            tempstr += '0'
    return tempstr

def Buildv(tree, path, filename, params):
    in_iw = params['IN_IW']
    in_dw = params['IN_DW']
    in_sign = params['IN_SIGN']
    out_iw = params['OUT_IW']
    out_dw = params['OUT_DW']
    out_sign = params['OUT_SIGN']

    str,w=get_str(tree,params)
    bits,max=get_bits(len(str))
    if len(str)==1:
        bits = 0

    f = open(path + '/' + filename + '.v', 'w')

    f.write('`timescale 1ns/1ps\n')
    f.write('module '+filename+'(\n')
    f.write('\tinput clk,rst,\n\tinput [{}:0] '.format(in_dw + in_iw - 1 + in_sign))
    for i in range(params['INPUT_NUM']):
        f.write('f{},'.format(i))
    f.write('\n\toutput reg [{}:0] out\n);\n\n'.format(out_dw + out_iw - 1 + out_sign))

    if bits != 0:
        f.write('wire [{}:0] chosen;\n'.format(bits-1))

    # 对每一位
    for i in range(bits):
        # 一个周期是多少，在这个周期内这个bit不会变，比如3个bits 000 第二位的周期就是 2^1 = 2
        scope=pow(2,i)
        tmp_str=""
        # 起点，比如倒数第二位起点是 10
        base=scope
        while base<len(str):        # base + scope这个范围内的bits为1
            if base+scope >= len(str):  # 但是base不能超上限，所以这里用tmp找出base之后连续多少个数都符合当前bit为1
                tmp=len(str)-base
            else:
                tmp=scope
            for offset in range(tmp):   # 既然base下面tmp个数的bit都为1，就开始遍历
                tmp_str+=str[base+offset]+' || '
            base+=2*scope               # 寻找下一个base
        tmp_str=tmp_str[0:-4]
        f.write('assign chosen[{}] = '.format(i)+ tmp_str +';\n')

    f.write('\nalways @ (posedge clk or negedge rst) begin\n')
    f.write('\tif (!rst) begin\n')
    f.write('\t\tout<= 0;\n')
    f.write('\tend\n')
    f.write('\telse begin\n')
    if bits != 0:
        f.write('\t\tcase(chosen)\n')
        for k in range(len(w)):
            t = w[k]
            for i in range(out_dw):
                t *= 2
            f.write('\t\t\t{}: out<={};\n'.format(k,get_binary(t,out_iw+out_dw,out_sign)))
        f.write('\t\t\tdefault: out<=0;\n')
        f.write('\t\tendcase\n')
    else:
        t = w[0]
        for i in range(out_dw):
            t *= 2
        f.write('\t\tout<={};\n'.format(get_binary(t,out_iw+out_dw,out_sign)))
    f.write('\tend\n\n')
    f.write('end\n')
    f.write('endmodule')
    f.close()

def Buildv2(tree, path, filename, params):
    in_iw = params['IN_IW']
    in_dw = params['IN_DW']
    in_sign = params['IN_SIGN']
    out_iw = params['OUT_IW']
    out_dw = params['OUT_DW']
    out_sign = params['OUT_SIGN']

    str,w=get_str(tree,params)

    f = open(path + '/' + filename + '.v', 'w')
    f.write('`timescale 1ns/1ps\n')
    f.write('module '+filename+' (\n')

    f.write('\tinput clk,rst,\n\tinput [{}:0] '.format(in_dw+in_iw-1+in_sign))
    for i in range(params['INPUT_NUM']):
        f.write('f{},'.format(i))
    f.write('\n\toutput reg [{}:0] out\n);\n\n'.format(out_dw+out_iw-1+out_sign))

    f.write('wire ')
    for i in range(len(str)):
        if i==0:
            f.write('path{}'.format(i))
        else:
            f.write(' ,path{}'.format(i))
    f.write(';\n')
    for i in range(len(str)):
        f.write('assign path{} = '.format(i) + str[i] + ';\n')

    f.write('\nalways @ (posedge clk or negedge rst) begin\n')
    f.write('\tif (!rst) begin\n')
    f.write('\t\tout<= 0;\n')
    f.write('\tend\n')
    f.write('\telse begin\n')
    f.write('\t\tcase({')
    for i in range(len(str)):
        if i==0:
            f.write('path{}'.format(i))
        else:
            f.write(' ,path{}'.format(i))
    f.write('})\n')
    for k in range(len(w)):
        temps = get_onehot(k,len(w))
        t = w[k]
        for i in range(out_dw):
            t *= 2
        f.write('\t\t\t{}\'b{}: out<={};\n'.format(len(w),temps,get_binary(t, out_iw + out_dw,out_sign)))
    f.write('\t\t\tdefault: out<=0;\n')
    f.write('\t\tendcase\n')
    f.write('\tend\n\n')
    f.write('end\n')
    f.write('endmodule')
    f.close()

def reversal3(node,str_set,w_set,str_now,params):
    in_iw = params['IN_IW']
    in_dw = params['IN_DW']
    in_sign = params['IN_SIGN']
    if node.leaf:
        if len(str_now)==0:
            w_set.append(node.w)
            str_set.append('')
            return
        str_tmp="("+str_now[0]
        i=1
        while i< len(str_now):
            str_tmp=str_tmp+" && "+str_now[i]
            i+=1
        str_tmp = str_tmp+")"
        str_set.append(str_tmp)
        w_set.append(node.w)
    else:
        num = node.val
        for i in range(in_dw):
            num *= 2
        str="(f{} < {})".format(node.f_id,get_binary(num,in_iw+in_dw,in_sign))
        str_now.append(str)
        reversal3(node.left,str_set,w_set,str_now,params)
        str_now.pop()
        str="(!(f{} < {}))".format(node.f_id,get_binary(num,in_iw+in_dw,in_sign))
        str_now.append(str)
        reversal3(node.right,str_set,w_set,str_now,params)
        str_now.pop()

def get_str3(tree,params):
    str_set=[]
    str_now=[]
    w_set=[]
    reversal3(tree,str_set,w_set,str_now,params)
    return str_set,w_set

def Buildv3(tree, path, filename, params):
    in_iw = params['IN_IW']
    in_dw = params['IN_DW']
    in_sign = params['IN_SIGN']
    out_iw = params['OUT_IW']
    out_dw = params['OUT_DW']
    out_sign = params['OUT_SIGN']

    str,w=get_str3(tree,params)

    f = open(path + '/' + filename + '.v', 'w')
    f.write('`timescale 1ns/1ps\n')
    f.write('module '+filename+' (\n')

    f.write('\tinput clk,rst,\n\tinput [{}:0] '.format(in_dw+in_iw-1+in_sign))
    for i in range(params['INPUT_NUM']):
        f.write('f{},'.format(i))
    f.write('\n\toutput reg [{}:0] out\n);\n\n'.format(out_dw+out_iw-1+out_sign))

    f.write('wire ')
    for i in range(len(str)):
        if i==0:
            f.write('path{}'.format(i))
        else:
            f.write(' ,path{}'.format(i))
    f.write(';\n')
    for i in range(len(str)):
        f.write('assign path{} = '.format(i) + str[i] + ';\n')

    f.write('\nalways @ (posedge clk or negedge rst) begin\n')
    f.write('\tif (!rst) begin\n')
    f.write('\t\tout<= 0;\n')
    f.write('\tend\n')
    f.write('\telse begin\n')
    f.write('\t\tcase({')
    for i in range(len(str)):
        if i==0:
            f.write('path{}'.format(i))
        else:
            f.write(' ,path{}'.format(i))
    f.write('})\n')
    for k in range(len(w)):
        temps = get_onehot(k,len(w))
        t = w[k]
        for i in range(out_dw):
            t *= 2
        f.write('\t\t\t{}\'b{}: out<={};\n'.format(len(w),temps,get_binary(t, out_iw + out_dw,out_sign)))
    f.write('\t\t\tdefault: out<=0;\n')
    f.write('\t\tendcase\n')
    f.write('\tend\n\n')
    f.write('end\n')
    f.write('endmodule')
    f.close()

######################################################################

def Build_adder(width,file_path):
    f = open(file_path,'w')
    f.write('`timescale 1ns / 1ps\n')
    f.write('module adder_{} (\n'.format(width))
    f.write('\tinput wire [{0}:0] add1,\n\
    input wire [{0}:0] add2,\n\
    input wire cin,\n\
    output wire [{0}:0] sum,\n\
    output wire cout\n);\n'.format(width-1))
    f.write('\twire[{}:0] xor_, and_, carry;\n'.format(width-1))
    f.write('\tassign xor_ = add1 ^ add2;\n')
    f.write('\tassign and_ = add1 & add2;\n')
    carry_str = 'cin'
    for i in range(width):
        if i!=0:
            carry_str = 'and_[{0}] | (xor_[{0}] & ({1}))'.format(i,carry_str)
        else:
            carry_str = 'and_[{0}] | (xor_[{0}] & {1})'.format(i,carry_str)
        f.write('\tassign carry[{}] = {};\n'.format(i,carry_str))
    f.write('\tassign sum[{0}:1] = xor_[{0}:1] ^ carry[{1}:0];\n'.format(width-1,width-2))
    f.write('\tassign sum[0] = xor_[0] ^ cin;\n')
    f.write('\tassign cout = carry[{}];\n'.format(width-1))
    f.write('endmodule\n')
    f.close()


def Build_adder_without_cin(width,file_path):
    f = open(file_path,'w')
    f.write('`timescale 1ns / 1ps\n')
    f.write('module adder_{} (\n'.format(width))
    f.write('\tinput wire [{0}:0] add1,\n\
    input wire [{0}:0] add2,\n\
    output wire [{0}:0] sum\n);\n'.format(width-1))
    f.write('\twire[{}:0] xor_, and_, carry;\n'.format(width-1))
    f.write('\tassign xor_ = add1 ^ add2;\n')
    f.write('\tassign and_ = add1 & add2;\n')
    carry_str = ''
    for i in range(width):
        if i!=0:
            carry_str = 'and_[{0}] | (xor_[{0}] & ({1}))'.format(i,carry_str)
        else:
            carry_str = 'and_[{0}]'.format(i,carry_str)
        f.write('\tassign carry[{}] = {};\n'.format(i,carry_str))
    f.write('\tassign sum[{0}:1] = xor_[{0}:1] ^ carry[{1}:0];\n'.format(width-1,width-2))
    f.write('\tassign sum[0] = xor_[0];\n')
    f.write('endmodule\n')
    f.close()

######################################################################

def Build_pipeline(input_width,input_num,file_path):
    cur_num = input_num
    cur_level = 0
    f = open(file_path,'w')
    f.write('`timescale 1ns / 1ps\n')
    f.write('module pipeline_{}_{} (\n'.format(input_width,input_num))
    f.write('\tinput clk,rst,\n')
    f.write('\tinput wire [{}:0] '.format(input_width-1))
    for i in range(input_num):
        f.write('in{}_{},'.format(cur_level,i))
    f.write('\n\toutput reg [{}:0] out\n);\n'.format(input_width-1))

    cur_level += 1

    always_str = []
    while(cur_num >= 2):
        pair_num = cur_num//2
        odd = cur_num % 2

        f.write('\twire [{}:0] '.format(input_width-1))
        for i in range(pair_num):
            if i == 0:
                f.write('wire{}_{}'.format(cur_level,i))
            else:
                f.write(',wire{}_{}'.format(cur_level,i))

        if odd == 1:
            f.write(',wire{}_{}'.format(cur_level,pair_num))
        f.write(';\n')

        f.write('\treg [{}:0] '.format(input_width-1))
        for i in range(pair_num):
            if i == 0:
                f.write('in{}_{}'.format(cur_level,i))
            else:
                f.write(',in{}_{}'.format(cur_level,i))
        if odd == 1:
            f.write(',in{}_{}'.format(cur_level,pair_num))
        f.write(';\n')

        for i in range(pair_num):
            f.write('\tadder_{} adder{}_{}(in{}_{},in{}_{},wire{}_{});\n'.format(\
                input_width,cur_level,i,cur_level-1,2*i,cur_level-1,2*i+1,cur_level,i))

        if cur_num != 2:
            for i in range(pair_num):
                always_str.append('in{0}_{1} <= wire{0}_{1};'.format(cur_level,i))
        else:
            always_str.append('out <= wire{0}_{1};'.format(cur_level,i))

        if odd==1:
            always_str.append('in{}_{} <= in{}_{};'.format(cur_level,pair_num,cur_level-1,cur_num-1))

        f.write('\n')
        cur_num = pair_num + odd
        cur_level += 1

    f.write('\talways @ (posedge clk or negedge rst) begin\n\
        if (!rst) begin\n\
            out<=0;\n\
        end\n\
        else begin\n')
    for line in always_str:
        f.write('\t\t\t'+line+'\n')
    f.write('\t\tend\n\tend\nendmodule')
    f.close()

######################################################################

def Build_top(input_width,input_num,output_width,tree_num,file_path):
    f = open(file_path,'w')
    f.write('`timescale 1ns / 1ps\n')
    f.write('module top (\n')
    f.write('\tinput clk,rst,\n')
    f.write('\tinput wire [{}:0] '.format(input_width-1))
    for i in range(input_num):
        f.write('f{},'.format(i))
    f.write('\n\toutput wire [{}:0] out\n);\n'.format(output_width-1))

    f.write('\twire [{}:0] '.format(output_width-1))
    for i in range(tree_num):
        if i == 0:
            f.write('out_{:03}'.format(i))
        else:
            f.write(',out_{:03}'.format(i))
    f.write(';\n')
    for i in range(tree_num):
        f.write('\ttree_{:03} tree{}(clk,rst'.format(i,i))
        for j in range(input_num):
            f.write(',f{}'.format(j))
        f.write(',out_{:03});\n'.format(i))

    f.write('\tpipeline_{}_{} adder(clk,rst'.format(output_width,tree_num))
    for j in range(tree_num):
        f.write(',out_{:03}'.format(j))
    f.write(',out);\n')
    f.write('endmodule')
    f.close()


######################################################################

def Build_xdc(inputnum, inputwidth, outputwidth, pin_file,file_path):
    pinfile = open(pin_file,'r')
    pins = pinfile.readlines()

    f = open(file_path,'w')
    f.write('set_property PACKAGE_PIN AU30 [get_ports clk]\n')
    f.write('create_clock -name sysclk -period 10 [get_ports clk]\n')
    f.write('set_property IOSTANDARD LVCMOS18 [get_ports clk]\n')

    f.write('set_property PACKAGE_PIN {} [get_ports rst]\n'.format(pins[0][0:-1]))
    f.write('set_property IOSTANDARD LVCMOS18 [get_ports rst]\n')

    idx = 1
    for i in range(outputwidth):
        f.write('set_property PACKAGE_PIN {} [get_ports {{out[{}]}}]\n'.format(pins[idx][0:-1],i))
        f.write('set_property IOSTANDARD LVCMOS18 [get_ports {{out[{}]}}]\n'.format(i))
        idx += 1
    
    for i in range(inputnum):
        for j in range(inputwidth):
            f.write('set_property PACKAGE_PIN {} [get_ports {{f{}[{}]}}]\n'.format(pins[idx][0:-1],i,j))
            f.write('set_property IOSTANDARD LVCMOS18 [get_ports {{f{}[{}]}}]\n'.format(i,j))
            idx += 1
            if idx >= len(pins):
                f.close()
                f = open(file_path,'w')
                f.write('need more pins')
                f.close()
                return


######################################################################

def Builder(trees, path, params, name):
    bar = Bar(name, max=len(trees), fill='#', suffix='%(percent)d%%')

    if os.path.exists(path) == False:
        os.mkdir(path)
    if os.path.exists(path+'_onehot') == False:
        os.mkdir(path+'_onehot')
    if os.path.exists(path+'_not') == False:
        os.mkdir(path+'_not')
    
    for index, tree in enumerate(trees):
        Buildv(tree, path, 'tree_{:03d}'.format(index), params)
        Buildv2(tree, path+'_onehot', 'tree_{:03d}'.format(index), params)
        Buildv3(tree, path+'_not', 'tree_{:03d}'.format(index), params)
        bar.next()
    print('')
