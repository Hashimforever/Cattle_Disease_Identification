import os, sys
from os import listdir
from os.path import join, isfile
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pdb
import glob

dd = pdb.set_trace

image = "image not found"

def read_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    return img


def read_for_train_img(imagName):
    global image
    for img in glob.glob(ForTrain + imagName + "*.*"):
        print(img)
        image = img
        if img is not None:
            break
    return image


class ImageList(data.Dataset):
    def __init__(self, list_file, transform=None, is_train=True,
                 img_shape=None):
        
        if img_shape is None:
            img_shape = [224, 224]
        #print('Total Number of images: %d ' % len(list_file))
        self.img_list = list_file
        self.transform = transform
        self.is_train = is_train
        self.img_shape = img_shape
        self.transform_img = transforms.Compose([self.transform])
        print(self.img_shape)

    def __getitem__(self, index):
        
        img1_path = self.img_list[index]
        token = img1_path.split('_')
        #print(token)
        disease = int(token[4])
        #view1 = int(token[6][5])
        img1 = read_img(img1_path)
    
        if self.transform_img is not None:
            img1 = self.transform_img(img1)  # [0,1], c x h x w
            
      
        return  img1, disease

    def __len__(self):
        return len(self.img_list)
        
    
#
 #if name == 'main':
#     image_dir = "C:/Users/gate/2021/Dataset/TomatoTrain/"
#
#     imgFiles_train = [
#              join(image_dir, fn)  # Create full paths to images
#              for fn in listdir(image_dir)  # For each item in the image folder
#              if isfile(join(image_dir, fn))  # If the item is indeed a file
#                 and fn.lower().endswith(('.png', '.jpg'))
#              # Which ends with an image suffix (can add more types here if needed)
#          ]
#     ba =1
#     trainImage = imgFiles_train
#     train = torch.utils.data.DataLoader(
#          ImageList(trainImage, transform=transforms.Compose([
#              transforms.ToTensor(),
#              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
#          batch_size=ba, shuffle=True,
#          num_workers=ba, pin_memory=True,drop_last=True)
#
    # main()
    # for i,  (disease1,img1) in  enumerate(train):
     #       print(disease1)
      #      print(img1)
       #     dd = pdb.set_trace()
#
#