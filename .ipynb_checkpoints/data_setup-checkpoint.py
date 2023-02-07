import os
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

torch.manual_seed(40)
def create_dataset(train_path, test_path, batch_size, ):
    
    transformer = transforms.Compose([
                                      transforms.RandomRotation(15),
                                      transforms.RandomHorizontalFlip(0.2),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      transforms.ToTensor()                                   
                                      ])
    
    #Create dataset from Image folder
    train_data = datasets.ImageFolder("defect_dataset/casting_data/casting_data/train",transform=transformer)
    test_data = datasets.ImageFolder("defect_dataset/casting_data/casting_data/test",transform=transformer)
    #Create loader
    train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
    teat_loader = DataLoader(test_data,batch_size=32)
    classes = train_data.classes
    return train_loader, teat_loader, classes
