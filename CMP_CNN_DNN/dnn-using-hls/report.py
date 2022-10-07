import pandas as pd
import time
from script import report
if __name__=='__main__':
    dict = {"rain":60,
            #  "mobile":20,
            #  "bank"	:15,
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
        # syn_impl(dataset)
        reports[i]= report(dataset)
    
    df = pd.DataFrame.from_dict(reports,orient='index')
    timestamp  = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    df.to_excel(f'DNN实验结果_{timestamp}.xlsx',index=False)