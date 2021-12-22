# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from flyai.utils.log_helper import train_log
from path import MODEL_PATH,DATA_PATH
import torch.nn as nn
from CNN import  CNN
from torch.autograd import Variable
from torch import optim
from torchvision import transforms
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from preprocess_data import MyDataset
from PIL import Image
import numpy as np
import torch.utils.data as Data
'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
args = parser.parse_args()

class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("WaterMeterNumberRecognition")

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''

        data_tf = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        self.train_data = MyDataset(root=DATA_PATH, datatxt='train.csv', transform=data_tf)



    def train(self):

        '''
        训练模型，必须实现此方法
        :return:
        '''
        device=torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device('cuda:0')

        net = CNN()
        net.to(device)

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), 1e-3)
        optimizer = optim.Adam(net.parameters(), 1e-3)
        nums_epoch =args.EPOCHS
        self.trainIter = Data.DataLoader(dataset=self.train_data, batch_size=args.BATCH, shuffle=True)
        losses = []
        acces = []

        for epoch in range(nums_epoch):
            # TRAIN
            train_loss = 0
            train_acc = 0
            net = net.train()
            for img, label in self.trainIter:
                # img = img.reshape(img.size(0),-1)
                img = Variable(img).to(device)
                label = Variable(label).to(device)

                # forward
                out = net(img)
                loss = criterion(out, label)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss
                train_loss += loss.item()
                # accuracy
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / img.shape[0]
                train_acc += acc

            losses.append(train_loss / len(self.trainIter))
            acces.append(train_acc / len(self.trainIter))

            # PRINT IN EVERYEPOCH
            print('Epoch {} Train Loss {} Train  Accuracy {}'.format(
                epoch + 1, train_loss / len(self.trainIter), train_acc / len(self.trainIter)))

            train_log(train_loss=train_loss, train_acc=train_acc)

        PATH = os.path.join(MODEL_PATH,'waterNumber_net.pth')
        torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()