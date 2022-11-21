import torch
from torch import nn
import torchvision
from torch.nn import functional as F
import timm
#pip install timm
from utils.sam import SAM



class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net=timm.create_model('efficientnetv2_rw_t',pretrained =True, num_classes=2)
        #input_size=(3, 224, 224), test_input_size=(3, 288, 288), pool_size=(7, 7), crop_pct=1.0
        self.cel=nn.CrossEntropyLoss()
        self.optimizer=SAM(self.parameters(),torch.optim.SGD,lr=0.001, momentum=0.9)
        
        

    def forward(self,x):
        x=self.net(x)
        return x
    
    def training_step(self,x,target):
        for i in range(2):
            pred_cls=self(x)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            
            loss=loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        return pred_first