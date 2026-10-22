"""BUỔI 3 — ĐỀ BÀI TRÊN LỚP: CNN hai block trên FashionMNIST.

Tập train=6000, validation=1200 lấy từ TRAIN chính thức, seed=42.
Không sử dụng tập TEST trong quá trình thực hành trên lớp.
Hãy điền TODO; 
Chạy: python bai03_class_cnn_fashionmnist.py
Cần: torch, torchvision, numpy, matplotlib.
"""
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED, EPOCHS, BATCH_SIZE = 42, 5, 64
LR, TRAIN_SIZE, VAL_SIZE = 1e-3, 6000, 1200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_data():
    all_train = datasets.FashionMNIST('./data', train=True, download=True,
                                     transform=transforms.ToTensor())
    ids = torch.randperm(len(all_train), generator=torch.Generator().manual_seed(SEED)).tolist()
    return Subset(all_train, ids[:TRAIN_SIZE]), Subset(all_train, ids[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE])

def make_loaders(train_ds, val_ds):
    tr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED), num_workers=0)
    va = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    return tr, va

class CNN2Blocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc2(self.relu(self.fc1(x)))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),
                                 nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x): return self.net(x)

def print_shapes(model):
    """TODO 1: In shape sau conv1, pool1, conv2, pool2, trước Linear với input (1,1,28,28)."""
    # Gợi ý: x = torch.zeros(1,1,28,28); model.conv1(x); model.relu(...); model.pool(...)
    raise NotImplementedError('TODO 1 — in đúng 5 shape')

def train_one_epoch(model, train_loader, optimizer, criterion):
    """TODO 2: model.train(), zero_grad, forward, loss, backward, step; trả train loss trung bình."""
    raise NotImplementedError('TODO 2 — huấn luyện một epoch')

@torch.inference_mode()
def validate(model, val_loader, criterion):
    """TODO 3: model.eval(); trả (validation_loss, validation_accuracy) theo SỐ MẪU."""
    raise NotImplementedError('TODO 3 — validation')

def main():
    set_seed()
    train_ds, val_ds = make_data()
    print('Device:', DEVICE, '| train:', len(train_ds), '| val:', len(val_ds))
    print_shapes(CNN2Blocks())
    for name, cls in [('MLP', MLP), ('CNN', CNN2Blocks)]:
        set_seed()
        model = cls().to(DEVICE)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_loader, val_loader = make_loaders(train_ds, val_ds)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        print(f'\\n{name} | trainable_params={params}')
        for epoch in range(1, EPOCHS+1):
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            _, accuracy = validate(model, val_loader, criterion)
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={accuracy:.2%}, time={elapsed:.2f}s')
    # TODO 4: Lập bảng so sánh số tham số, final val_accuracy, giây/epoch trung bình.
    # TODO 5: Viết nhận xét 3–5 câu dựa trên kết quả thực tế.

if __name__ == '__main__':
    main()
