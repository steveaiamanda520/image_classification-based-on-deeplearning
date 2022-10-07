#############################################################
############  编写人：常泽钰
############  时间 ： 2022/9/23
############  功能 ：实现明星的图片分类
############  数据格式：
'''
                    --dataset
                        --train
                            --0
                                --**.jpg
                                --.......
                            --1
                            --2
                            --3
                        --val
'''
#############################################################
batch_size = 8         #设置批次大小
learning_rate = 1e-4        #设置学习率
epoches = 2                 #设置训练的次数
num_of_classes=10           #要分的类别个数


import torch
import os
from torch.utils import data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
trainpath = './dataset/train/'
valpath = './dataset/val/'

#数据增强的方式
traintransform = transforms .Compose([
 transforms .RandomRotation (20),               #随机旋转角度
 transforms .ColorJitter(brightness=0.1),     #颜色亮度
 transforms .Resize([224, 224]),               #设置成224×224大小的张量
 transforms .ToTensor(),                        # 将图⽚数据变为tensor格式
# transforms.Normalize(mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225]),
])


valtransform = transforms .Compose([
 transforms .Resize([224, 224]),
 transforms .ToTensor(), # 将图⽚数据变为tensor格式
])



trainData = dsets.ImageFolder (trainpath, transform =traintransform ) # 读取训练集，标签就是train⽬录下的⽂件夹的名字，图像保存在格⼦标签下的⽂件夹⾥
valData = dsets.ImageFolder (valpath, transform =valtransform )       #读取演正剧
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)     #将数据集分批次  并打乱顺序
valLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=batch_size, shuffle=False)          #将测试集分批次并打乱顺序

test_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(trainpath))])        #计算  训练集和测试集的图片总数
train_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(valpath))])

import numpy as np

import torchvision.models as models
model = models.resnet34(pretrained=True) #pretrained表⽰是否加载已经与训练好的参数
model.fc = torch.nn.Linear(512, num_of_classes) #将最后的fc层的输出改为标签数量（如3）,512取决于原始⽹络fc层的输⼊通道
#model = model.cuda() # 如果有GPU，⽽且确认使⽤则保留；如果没有GPU，请删除
#from MyLoss import Loss
#criterion=Loss("mean")
criterion = torch.nn.CrossEntropyLoss() # 定义损失函数



#optimizer = torch.optim.Adam(model.parameters (), lr=learning_rate )
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # 定义优化器


from torch.autograd import Variable
#定义训练的函数
def train(model, optimizer, criterion):
    model.train()
    total_loss = 0
    train_corrects = 0
    for i, (image, label) in enumerate (trainLoader):
         #image = Variable(image.cuda()) # 同理
         #label = Variable(label.cuda()) # 同理
         #print(i,image,label)
         optimizer.zero_grad ()
         target = model(image)
         loss = criterion(target, label)
         loss.backward()
         optimizer.step()
         total_loss += loss.item()
         max_value , max_index = torch.max(target, 1)
         pred_label = max_index.cpu().numpy()
         true_label = label.cpu().numpy()
         train_corrects += np.sum(pred_label == true_label)
    return total_loss / float(len(trainLoader)), train_corrects / train_sum


testLoader=valLoader
#定义测试的函数
def evaluate(model, criterion):
    model.eval()
    corrects = eval_loss = 0
    with torch.no_grad():
        for image, label in testLoader:
            #image = Variable(image.cuda()) # 如果不使⽤GPU，删除.cuda()
            #label = Variable(label.cuda()) # 同理
            pred = model(image)
            loss = criterion(pred, label)
            eval_loss += loss.item()
            max_value, max_index = torch.max(pred, 1)
            pred_label = max_index.cpu().numpy()
            true_label = label.cpu().numpy()
            corrects += np.sum(pred_label == true_label)
    return eval_loss / float(len(testLoader)), corrects, corrects / test_sum

#torch.save(model,"./resnet1.pt")

for i in range(epoches):
    print("第{}个epoch".format(i+1))
    train_loss,train_acc=train(model,optimizer,criterion)
    print("train_loss: {}   train_acc: {}\n".format(train_loss,train_acc))
    test_loss,test_correct,test_acc=evaluate(model,criterion)
    print("test_loss: {}    test_correct:{}    test_acc:{}".format(test_loss,test_correct,test_acc))



torch.save(model,"./resnet34_2.pt")#保存模型，第二个参数是保存的路径


