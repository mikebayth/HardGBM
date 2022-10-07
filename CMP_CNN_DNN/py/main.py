from DNN import DNN
from Calc_bit import d2b
import time
import torch

def run():
    beginTime = time.perf_counter()

    modul = DNN()
    modul.train(20,"hospital")

    torch.save(modul.state_dict(), "DNN-hospital.pth")
    
    # modul.load_state_dict(torch.load("CNN.pth"))

    # for param in modul.named_parameters():
    #     if(param[0] == 'conv1.weight') or (param[0] == 'conv2.weight') or (param[0] == 'conv3.weight'):
    #         f = open(param[0] + ".coe", "w")
    #         data = ';\nmemory_initialization_radix = 2;\nmemory_initialization_vector=\n'
    #         f.writelines(data)
    #         para = param[1].data.detach().numpy()
    #
    #         for m in range(para.shape[2]):
    #             for n in range(para.shape[3]):
    #                 for i in range(para.shape[0]):
    #                     for j in range(para.shape[1]):
    #                         data = d2b(para[i][j][m][n], 2, 13)
    #                         f.writelines(data)
    #                 f.writelines('\n')
    #         f.writelines(';')
    #         f.close()
    #
    #     if (param[0] == 'conv1.bias') or (param[0] == 'conv2.bias') or (param[0] == 'conv3.bias'):
    #         f = open(param[0] + ".coe", "w")
    #         data = ';\nmemory_initialization_radix = 2;\nmemory_initialization_vector=\n'
    #         f.writelines(data)
    #         para = param[1].data.detach().numpy()
    #
    #         for m in range(para.shape[0]):
    #                 data = d2b(para[m], 5, 26)
    #                 f.writelines(data)
    #                 f.writelines('\n')
    #         f.writelines(';')
    #         f.close()
    #
    #     if (param[0] == 'fc1.weight') or (param[0] == 'fc2.weight'):
    #         f = open(param[0] + ".coe", "w")
    #         data = ';\nmemory_initialization_radix = 2;\nmemory_initialization_vector=\n'
    #         f.writelines(data)
    #         para = param[1].data.detach().numpy()
    #
    #         for n in range(para.shape[1]):
    #             for m in range(para.shape[0]):
    #                 data = d2b(para[m][n], 2, 13)
    #                 f.writelines(data)
    #             f.writelines('\n')
    #         f.writelines(';')
    #         f.close()
    #
    #

    endTime = time.perf_counter()
    print("Time:"+str(endTime-beginTime))

if __name__ == '__main__':
    run()