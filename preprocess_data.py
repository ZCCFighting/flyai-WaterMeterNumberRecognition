import os
import  cv2 as cv
from path import  DATA_PATH
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None):
        super(MyDataset, self).__init__()
        # with open(f'{root}/{datatxt}', 'r') as f:
        #         #     imgs = []
        #         #     # 读取csv信息到imgs列表
        #         #     for path, label in map(lambda line: line.rstrip().split(','), f):
        #         #         imgs.append((path, int(label)))
        csv_path=os.path.join(root, "WaterMeterNumberRecognition/"+datatxt)
        image_path = os.path.join(root, "WaterMeterNumberRecognition")
        data_form = pd.read_csv(csv_path)
        image_list = data_form['image_path']
        label_list = data_form['label']
        imgs = []
        for n in range(len(image_list)):
            image_temppath = os.path.join(image_path, image_list[n])
            label_n = label_list[n].split(',')
            image = cv.imread(image_temppath)
            height, width, _ = image.shape
            num = width // 5
            for i in range(5):
                image_temp = image[:, i * num:(i + 1) * num:, :]

                image_temp = cv.resize(image_temp, (28, 28))

                file_name = str(n) + "_" + str(i) + ".jpg"
                file_path = os.path.join(image_path, file_name)
                cv.imwrite(file_path, image_temp)
                imgs.append((file_path, int(label_n[i])))
        self.imgs = imgs
        self.transform = transform if transform is not None else lambda x: x


    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.transform(Image.open(path).convert('L'))
        return img, label


    def __len__(self):
        return len(self.imgs)



# def resize_image(image):
#     height,width,_=image.shape
# save_path=os.path.join(DATA_PATH, "WaterMeterNumberRecognition/res")
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# label_res=os.path.join(DATA_PATH,"label.txt")
# f=open(label_res,'w')
# def process_data():
#     image_path = os.path.join(DATA_PATH, "WaterMeterNumberRecognition")
#     csv_path = os.path.join(DATA_PATH, "WaterMeterNumberRecognition/train.csv")
#     data_form = pd.read_csv(csv_path)
#     image_list = data_form['image_path']
#     label_list = data_form['label']
#     res_image=[]
#     res_label=[]
#
#     for n in range(len(image_list)):
#         image_temppath=os.path.join(image_path,image_list[n])
#         label_n=label_list[n].split(',')
#         image=cv.imread(image_temppath)
#         height,width,_=image.shape
#         num=width//5
#         for i in range(5):
#             image_temp=image[:,i*num:(i+1)*num:,:]
#             image_temp=cv.resize(image_temp,(28,28))
#             file_name=str(n)+"_"+str(i)+".jpg"
#             file_path=os.path.join(save_path,file_name)
#             cv.imwrite(file_path,image_temp)
#             line=str(file_path+" "+label_n[i]+"\n")
#             res_label.append(line)
#             f.writelines(line)
#     f.close()
# def resize_image(img,size=28):

