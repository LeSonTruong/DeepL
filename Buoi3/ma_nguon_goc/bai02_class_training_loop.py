"""Bài 2 - trên lớp: viết training loop rõ từng bước bằng FashionMNIST."""
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
from common import seed_everything,device,evaluate_classifier
seed_everything(42); dev=device()
tfm=transforms.ToTensor(); root='./data'
train_full=datasets.FashionMNIST(root,train=True,download=True,transform=tfm)
test_full=datasets.FashionMNIST(root,train=False,download=True,transform=tfm)
train=Subset(train_full,range(6000)); val=Subset(test_full,range(1000))
train_loader=DataLoader(train,64,shuffle=True); val_loader=DataLoader(val,256)
model=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10)).to(dev)
loss_fn=nn.CrossEntropyLoss(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
for epoch in range(1,6):
    model.train()
    for xb,yb in train_loader:
        xb,yb=xb.to(dev),yb.to(dev)
        optimizer.zero_grad()
        logits=model(xb)
        loss=loss_fn(logits,yb)
        loss.backward()
        optimizer.step()
    val_loss,val_acc=evaluate_classifier(model,val_loader,loss_fn)
    print(epoch,val_loss,val_acc)
