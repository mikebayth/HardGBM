import re
import os 
# vivado_path = "hardware_lgb/"

# outputfile = "hardware_report_lgb.txt"
def report(vivado_path,outputfile):
    dics = os.listdir(vivado_path)

    name = []
    util_LUT = []
    util_Reg = []
    power = []

    for dic in dics:
        name.append(dic)
        path = vivado_path + dic+ '/vivado_out/'
        power_path = path + 'place_power.rpt'
        util_path = path + 'place_utilization.rpt'

        with open(power_path,'r') as power_file:
            lines = power_file.readlines()
        for line in lines:
            if line.find('Total On-Chip Power (W)')>=0:
                ans = re.findall(r"\d+\.?\d*", line)
                temp = ans[0].strip()
                power.append(float(temp))
                break
        with open(util_path,'r') as util_file:
            lines = util_file.readlines()
        for line in lines:
            if line.find('| Slice LUTs')>=0:
                ans = re.findall('(\d+)', line)
                temp = ans[0].strip()
                util_LUT.append(int(temp))
                break
        for line in lines:
            if line.find('| Slice Registers')>=0:
                ans = re.findall('(\d+)', line)
                temp = ans[0].strip()
                util_Reg.append(int(temp))
                break


    f = open(outputfile,'w')
    i = 0
    while i < len(name):
        f.write('{}\n'.format(name[i]))
        f.write('LUTs\tFFs\tPower\n')
        f.write('{}\t{}\t{:.3f}\n'.format(util_LUT[i],util_Reg[i],power[i]))
        f.write('{}\t{}\t{:.3f}\n'.format(util_LUT[i+1],util_Reg[i+1],power[i+1]))
        f.write('{}\t{}\t{:.3f}\n'.format( (util_LUT[i]+util_LUT[i+1])//2 ,(util_Reg[i]+util_Reg[i+1])//2 ,(power[i]+power[i+1])/2))
        f.write('& {} & {} &{}\n\n'.format( (util_LUT[i]+util_LUT[i+1])//2 ,(util_Reg[i]+util_Reg[i+1])//2 ,int(((power[i]+power[i+1])*1000)//2)))

        i+=2
    f.close()
if __name__ == '__main__':
    vivado_dict = {
        "hardware_lgb/": "report_lgb.txt",
        "hardware_xgb/": "report_xgb.txt",
        "hardware_xgb_bucket/": "report_xgb_bucket.txt"
    }
    for k,v in vivado_dict.items():
        report(k,v)
