import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np


def picture_tensor(array1):#展示张量图片
    maxValue=array1.max()
    array1=array1*255/maxValue#normalize，将图像数据扩展到[0,255]
    img=np.uint8(array1)
    #print(img)
    img=img.transpose(1,2,0)
    #print(img.shape)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.rectangle(img, (int(224*label[0]),int(224*label[1])), (int(224*label[2]),int(224*label[3])), (255,0,0), 2)
    cv2.imshow("图",img)
    cv2.waitKey(0)


model=torch.load("./resnet34_2.pt")
model.eval()
transformer = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),  # 把PIL核np.array格式的图像转化为Tensor
        ])




#图片的路径
filename="./2.png"
img=Image.open(filename)
img=transformer(img)
picture_tensor(img)
img= img.unsqueeze(0)
pred=model(img)
#print(pred)
max_value, max_index = torch.max(pred, 1)
print("结果是:"+str(max_index.numpy()[0]))