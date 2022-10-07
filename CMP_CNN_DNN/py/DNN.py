import numpy as np
from Calc_bit import d2b

import torch

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from load_data import load_by_argv

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(17,256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,2)

        self.drop = nn.Dropout(0.2)



    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = F.relu(self.fc4(x))

        return x

    def train(self,num_epoch,dataname):
        print("Train:\n")

        # 导入并处理数据
        X, y, dic = load_by_argv(dataname)
        X = (X.values).astype(np.float32)

        # X = np.delete(X,[2],axis=1) # 如果选择customer数据集，第二列数据为nan，需要删除
        # X = np.delete(X, [8,9,10,11,12,18,19,20,21,35,51,52,53,70,71], axis=1) # 如果选择fraud数据集，需要删除相应nan的列

        # 写入某一张input
        # X_cnn = np.zeros([X.shape[0], 1, X.shape[1], X.shape[1]], dtype=np.float32)
        # for i in range(len(X)):
        #     for j in range(X.shape[1]):
        #         X_cnn[i][0][j] = X[i]
        #
        # img=X_cnn[0][0]
        # f = open("input.coe", "a")
        # data = ';\nmemory_initialization_radix = 2;\nmemory_initialization_vector=\n'
        # f.writelines(data)
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         data = d2b(img[i][j], 8, 7)
        #         f.writelines(data)
        #         f.writelines('\n')
        # f.writelines(';')
        # f.close()

        y = y.values.ravel()

        train_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=50)

        # 指定优化器
        optimizer=optim.SGD(self.parameters(),lr=0.01,momentum=0.5)
        # 指定损失函数
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epoch):
            num_correct = 0
            for idx,(data,target) in enumerate(train_loader):
                optimizer.zero_grad()
                target=target.long()
                output=self(data)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                _, pred = output.max(1)
                num_correct += (pred == target).sum().item()
                if idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, idx * len(data), len(train_loader.dataset),100. * idx / len(train_loader), loss.item()))

            print("epoch:{} accurancy:{}\n".format(epoch, num_correct / len(train_data)))





