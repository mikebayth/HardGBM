import numpy as np
from Calc_bit import d2b

import torch

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from load_data import load_by_argv

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Conv2d(1,4,kernel_size=3,stride=1)
        self.conv2=nn.Conv2d(4,8,kernel_size=3,stride=1)
        self.conv3=nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.fc1=nn.Linear(400,100)
        #self.bn1=nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(100, 2)



    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # x = F.relu(F.max_pool2d(self.conv3(x), 2))

        x = self.conv3(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def train(self,num_epoch,dataname):
        print("Train:\n")

        # 导入并处理数据
        X, y, dic = load_by_argv(dataname)
        X = (X.values).astype(np.float32)
        # X = np.delete(X, [8,9,10,11,12,18,19,20,21,35,51,52,53,70,71], axis=1)

        X_cnn = np.zeros([X.shape[0], 1, X.shape[1], X.shape[1]], dtype=np.float32)
        for i in range(len(X)):
            for j in range(X.shape[1]):
                X_cnn[i][0][j] = X[i]

        img=X_cnn[0][0]
        f = open("input.coe", "a")
        data = ';\nmemory_initialization_radix = 2;\nmemory_initialization_vector=\n'
        f.writelines(data)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                data = d2b(img[i][j], 8, 7)
                f.writelines(data)
                f.writelines('\n')
        f.writelines(';')
        f.close()

        y = y.values.ravel()

        train_data = TensorDataset(torch.from_numpy(X_cnn), torch.from_numpy(y))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=50)

        optimizer=optim.SGD(self.parameters(),lr=0.01,momentum=0.5)
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





