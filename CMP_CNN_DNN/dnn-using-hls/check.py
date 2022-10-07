import subprocess,os
from time import sleep
import threading
 
#检测进程
def check(findKey):
    p1 = subprocess.Popen(['ps','-ef'],stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['grep',findKey],stdin=p1.stdout,stdout=subprocess.PIPE)
    r = p2.stdout.readlines()
    # flag = False
    # print(len(r))
    # print(r)
    
 
    return len(r)>1
 
# #运行进程
# def run(cmd, out):
#     realCmd = 'nohup python /path/to/' + cmd + ' > ' + out + ' &'
#     os.system(realCmd)
 
# def thread1():
#     sleep(6000)
#     print('sleep over') 

# def thread2():
   
#     while True:
#         if check('vivado') :
#             print('vivado is running')
#             sleep(1800)
#         else:
#             os.system('python report.py')
#             break
if __name__ == '__main__':
    # t1 = threading.Thread(name='t1',target= thread1)
    # t2 = threading.Thread(name='t2',target= thread2)
    # t1.start()
    # t1.join()
    # t2.start()
    # t2.join()

    # sleep(10000)
    # print('sleep over')
    while True:
        if check('vivado') :
            print('vivado is running')
            sleep(1800)
        else:
            os.system('python report.py')
            break
            
