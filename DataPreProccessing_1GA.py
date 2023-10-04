 
import os, sys
from os import listdir
from os.path import join, isfile

import numpy as np
from PIL import Image
import random
from os.path import isfile, join
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import DataLoader
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

batch_size = 8


def preproccessing():
    

    # Define the data transforms
    transform_train = transforms.Compose([

          transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5,0.5))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5,0.5))
    ])

    
    # Load the dataset and split it into train and validation sets
    train_dataset = ImageFolder(root='E:/jupyter notebook/my_trying_fie/All_In_One/DatasetContainer/Processed_10000_FirstGA_other/train', transform=transform_train)
    num_samples = len(train_dataset)
    num_train_samples = int(num_samples * 0.9)
    num_valid_samples = num_samples - num_train_samples
    trainloader, validloader = random_split(train_dataset, [num_train_samples, num_valid_samples])

    # Define the data loaders
    train_loader = DataLoader(trainloader, batch_size=8, shuffle=False)
    #val_dataset = ImageFolder(root='E:/jupyter notebook/my_trying_fie/All_In_One/DatasetContainer/Processed_10000_FirstGA_other/train', transform=transform_train)
    valid_loader = DataLoader(validloader, batch_size=8, shuffle=False)

    print("Total Number of training and validation images processing per batches:",len(train_dataset))
    print("Total Number of training images processing per batches:",len(train_loader),len(valid_loader))


    test_dataset = datasets.ImageFolder(root='E:/jupyter notebook/my_trying_fie/All_In_One/DatasetContainer/Processed_10000_FirstGA_other/test', transform=transform_test)
    TestLoader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
    print("Total Number of images in Testing folder:",len(test_dataset))
    

    # Define the class labels
    class_labels = ['FMD', 'KCD', 'LD', 'RWD', 'WD']

    
    
    return train_loader, valid_loader, TestLoader
    # return  synth_loader

