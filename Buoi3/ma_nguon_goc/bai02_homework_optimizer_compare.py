"""Bài 2 - BTVN: so sánh SGD, SGD+momentum và Adam trên FashionMNIST."""
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
from common import seed_everything,device,evaluate_classifier
seed_everything(42); dev=device(); ds=datasets.FashionMNIST('./data',train=True,download=True,transform=transforms.ToTensor())
train=Subset(ds,range(8000)); loader=DataLoader(train,128,shuffle=True)
test=datasets.FashionMNIST('./data',train=False,download=True,transform=transforms.ToTensor()); val=DataLoader(Subset(test,range(1500)),256)
for name in ['sgd','momentum','adam']:
    model=nn.Sequential(nn.Flatten(),nn.Linear(784,128),nn.ReLU(),nn.Linear(128,10)).to(dev)
    if name=='sgd': opt=torch.optim.SGD(model.parameters(),lr=.05)
    elif name=='momentum': opt=torch.optim.SGD(model.parameters(),lr=.03,momentum=.9)
    else: opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn=nn.CrossEntropyLoss(); print('\n',name)
    for epoch in range(1,6):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
        print(epoch,evaluate_classifier(model,val,loss_fn))
