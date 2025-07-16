import torch
import torch.nn as nn
import os
from dataset import ImageNet
from torchvision import transforms
from omegaconf import  OmegaConf
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from CustomModel import CustomModel, ModelWarpper
import time

class Trainer():
  def __init__(self,config_path):
    config = OmegaConf.load(config_path)
    self.model_path = config.model_path
    self.data_path = config.data_folder
    self.batch_size = config.batch_size
    self.lr = config.lr
    self.epoch = config.epoch
    self.momentum = config.momentum
    self.weight_decay = config.weight_decay
    self.class_number = config.class_number
    self.resolution = config.resolution
    
    self.train_loader,self.test_loader = self.set_loader()
    self.model = torch.load(self.model_path).cuda()
    self.loss_func = torch.nn.CrossEntropyLoss()
    
    self.optimizor = optim.SGD(self.model.parameters(),
                          lr=self.lr,
                          momentum=self.momentum,
                          weight_decay=self.weight_decay)
    self.scheduler = StepLR(self.optimizor, step_size=20, gamma=0.1)

    
  def set_dataset(self):
    dataset_class = ImageNet
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = dataset_class(root=self.data_path,
                                  train=True,
                                  transform=train_transform)
    test_dataset = dataset_class(root=self.data_path,
                                 train=False,
                                 transform=test_transform)
    return train_dataset, test_dataset

  def set_loader(self):
    train_dataset, test_dataset = self.set_dataset() #both are poisioned dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=self.batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=self.batch_size, shuffle=True,
                                              num_workers=4, pin_memory=True)
    return train_loader, test_loader

  def train_one_epoch(self):
    '''Train one epoch'''
    self.model.train()
    loss = 0
    accuracy = 0 
    for index, data in enumerate(self.train_loader):
        images = data[0]
        labels = data[1]
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        output = self.model(images)
        loss = self.loss_func(output, labels)
        accuracy += torch.eq(output.argmax(1),labels.squeeze(dim=-1)).float().mean()
        
        self.optimizor.zero_grad()
        loss.backward()
        self.optimizor.step()
    accuracy = accuracy/index
    return loss, accuracy
        
  def train(self):
    print("Now starting training")
    start = time.perf_counter()
    for epoch in range(1, self.epoch + 1):
        # train for one epoch
        train_loss, train_acc = self.train_one_epoch()
        self.scheduler.step()
        print('epoch:',epoch,'loss:',train_loss.item(),'Training Accuracy:',train_acc.item())
        if epoch % 5 == 0:
          self.test()
    end = time.perf_counter()
    runTime = end-start
    print("Running Time:",runTime)
          
  def test(self):
    print("Now starting test")
    self.model.eval()
    accuracy = 0
    for index, data in enumerate(self.test_loader):
        images = data[0]
        labels = data[1]
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        output = self.model(images)
        accuracy += torch.eq(output.argmax(1),labels.squeeze(dim=-1)).float().mean()
    
    print('Test Accuracy:',accuracy.item()/index)
    
if __name__ == '__main__':
    config_path = './config/default.yaml'
    Trainer = Trainer(config_path=config_path)
    Trainer.train()
    
    
