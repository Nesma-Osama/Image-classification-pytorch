import torchvision.models as models
import torch.nn as nn
class VGG_Model(nn.Module):
    def __init__(self,output_class):
        super(VGG_Model,self).__init__()
        self.model=models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in self.model.features.parameters():
            param.requires_grad=False
        in_features=self.model.classifier[-1].in_features
        self.model.classifier[-1]=nn.Linear(in_features,output_class)
    def forward(self,x):
        return self.model(x)
    