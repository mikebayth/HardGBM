import os

path = './'
hard_paths = [p for p in os.listdir(path) if 'hardware_' in p and os.path.isdir(p)]
# print(hard_paths)
for hard_path in hard_paths:
    for vivado_proj in os.listdir(hard_path):
        # print(vivado_proj)
        vivado_tcl = f'{hard_path}/{vivado_proj}/run.tcl'
        os.system('cp run.tcl '+ vivado_tcl)
        # To avoid exhausting the CPU, uncomment the following two lines of code
        # import time 
        # time.slepp(30)
        os.system(f'cd {hard_path}/{vivado_proj} && nohup vivado -mode batch -source run.tcl &')