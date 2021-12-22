# -*- coding: utf-8 -*
# from flyai.framework import FlyAI
import cv2 as cv
from path import  MODEL_PATH
from  torch import  nn
from  PIL import  Image
from torchvision import transforms
import  torch
import numpy
import os
from CNN import CNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class resizeNormalize_id(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        # img.sub_(0.588).div_(0.193)
        return img
# class Prediction(FlyAI):
class Prediction():
# class Prediction():
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        self.model =CNN().to(device)
        # self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'waterNumber_net.pth')))
        model_path="./model/waterNumber_net.pth"
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":".\/data\/input\/image\/0.jpg"}
        :return: 模型预测成功之后返回给系统样例 {"label":"0,1,2,7,12"}
        '''
        self.model.eval()
        transform_valid = transforms.Compose([
            transforms.Resize((28, 28), interpolation=2),transforms.ToTensor()
        ])
        img = Image.open(image_path)

        width,height=img.size
        num=width//5
        res=""
        for i in range(5):
            img = cv.cvtColor(numpy.asarray(img), cv.COLOR_RGB2BGR)
            img_temp=img[:,i*num:(i+1)*num,:]
            img_temp=Image.fromarray(cv.cvtColor(img_temp,cv.COLOR_BGR2RGB)).convert("L")
            transformer = resizeNormalize_id((28,28))
            img_ = transformer(img_temp).unsqueeze(0)
            img_ = img_.to(device)
            output=self.model(img_).tolist()[0]
            # print(output)
            prediction = output.index(max(output))
            res+=str(prediction)+","
        res=res[0:-1]
        return {"label":res}