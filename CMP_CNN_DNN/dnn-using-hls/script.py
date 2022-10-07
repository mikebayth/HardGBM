import shutil
import os
import sys
import re
import pandas as pd
import time 
def modify_line(dataset):
    # cpp
    from_file = open("dnn.cpp") 
    line = from_file.readline()
    line = f'#include "{dataset}/weights.h"\n'
    to_file = open("dnn.cpp",mode="w")
    to_file.write(line)
    shutil.copyfileobj(from_file, to_file)

    # hls tcl
    from_file = open("run_hls.tcl") 
    line = from_file.readline()
    line = f'open_project {dataset}-proj \n'
    to_file = open("run_hls.tcl",mode="w")
    to_file.write(line)
    shutil.copyfileobj(from_file, to_file)

    #vivado tcl

def report(dataset):
    res_dict = {}
    res_dict["Dataset"] = dataset
    vivado_path = f'{dataset}-proj/solution1/impl/verilog/vivado_out/'
    power_path = vivado_path + 'place_power.rpt'
    util_path = vivado_path + 'place_utilization.rpt'

    with open(util_path,'r') as util_file:
        lines = util_file.readlines()
    for line in lines:
        if line.find('| Slice LUTs')>=0:
            ans = re.findall('(\d+)', line)
            temp = ans[0].strip()
            res_dict['LUTs'] = temp

            break
    for line in lines:
        if line.find('| Slice Registers')>=0:
            ans = re.findall('(\d+)', line)
            temp = ans[0].strip()
            res_dict['FFs'] = temp

            break
    for line in lines:
        if line.find('| Block RAM Tile')>=0:
            ans = re.findall('(\d+)', line)
            temp = ans[0].strip()
            res_dict['BRAMs'] = temp
            break

    with open(power_path,'r') as power_file:
        lines = power_file.readlines()

    for line in lines:
        if line.find('Device Static (W)')>=0:
            ans = re.findall(r"\d+\.?\d*", line)
            temp = ans[0].strip()
            res_dict['Power (W)'] = temp
            break
   
    return res_dict

def syn_impl(dataset):
    modify_line(dataset)
    os.system('nohup vivado_hls -f run_hls.tcl')
    os.system(f'cp run.tcl ./{dataset}-proj/solution1/impl/verilog/')
    os.system(f'cd ./{dataset}-proj/solution1/impl/verilog/ && nohup vivado -mode batch -source run.tcl &')


        
if __name__=='__main__':
    dict = {
                "rain":60,
                # "mobile":20,
                # "bank"	:15,
                # 'water'	:20,
                # 'customer':11,
                # 'rice'	:10,
                # 'income':14,
                # 'fraud':97,
                # 'heartDisease':21,
                # 'hospital':17
    }

    reports = {}
    for i, dataset in enumerate(dict.keys()) :
        syn_impl(dataset)
    #     reports[i]= report(dataset)
    
    # df = pd.DataFrame.from_dict(reports,orient='index')
    # timestamp  = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    # df.to_excel(f'DNN实验结果_{timestamp}.xlsx',index=False)

