import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

num_classes = 5

class Pre_ResNet18Model(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=5):
        super(Pre_ResNet18Model, self).__init__()

        # Load the pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=True)

        # Freeze the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer with a new one that has 512 output features
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 256)

        # Add a batch normalization layer and a new activation
        self.model.bn = nn.BatchNorm1d(256)
        self.model.relu = nn.ReLU(inplace=True)

        # Add a dropout layer with probability 0.5
        self.model.dropout = nn.Dropout(p=0.5)

        
        # Add a new fully connected layer with num_classes output classes and softmax activation
        self.model.fc2 = nn.Linear(256, num_classes)
        self.model.softmax = nn.Softmax(dim=1)

        
    def forward(self, x):
        x = self.model(x)
        return x


def Pre_ResNet18Model1(**kwargs):
    model = Pre_ResNet18Model(num_classes=5)
    return model
model=Pre_ResNet18Model1(num_classes=5)
#print("this model is Pre_ResNet18Model1")
#print(model)

