"""BUỔI 3 — ĐỀ BÀI VỀ NHÀ: biến thể CNN.

Mức cơ bản: baseline CNN, accuracy và confusion matrix trên test.
Mức khá: một biến thể kernel/channel. Nâng cao: BatchNorm/Dropout.
Nộp: code, bảng kết quả, learning curves, 5 ảnh đúng và 5 ảnh sai.

Chỉ chọn cấu hình theo validation; chỉ dùng official test để đánh giá cuối.
Điền các TODO dưới đây.
"""
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATASET = 'fashion'  # 'fashion' hoặc 'cifar'
SEED, EPOCHS, BATCH_SIZE = 42, 5, 64
LR, TRAIN_SIZE, VAL_SIZE = 1e-3, 6000, 1200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_splits():
    cls = datasets.FashionMNIST if DATASET == 'fashion' else datasets.CIFAR10
    full = cls('./data', train=True, download=True, transform=transforms.ToTensor())
    ids = torch.randperm(len(full), generator=torch.Generator().manual_seed(SEED)).tolist()
    train_ds = Subset(full, ids[:TRAIN_SIZE])
    val_ds = Subset(full, ids[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE])
    return train_ds, val_ds, cls

def make_loaders(train_ds, val_ds):
    tr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED), num_workers=0)
    va = DataLoader(val_ds, batch_size=256, num_workers=0)
    return tr, va

class CNN2Blocks(nn.Module):
    def __init__(self, in_channels=1, side=28, channels=16, kernel_size=3,
                 batchnorm=False, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels) if batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, 2*channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(2*channels) if batchnorm else nn.Identity()
        self.relu, self.pool = nn.ReLU(), nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2*channels*(side//4)**2, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        return self.fc2(self.drop(self.relu(self.fc1(x))))

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()*x.size(0); n += x.size(0)
    return total/n

@torch.inference_mode()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss_sum += criterion(logits, y).item()*x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        count += x.size(0)
    return loss_sum/count, correct/count

def experiments():
    """TODO 1: bổ sung ít nhất một cấu hình channel/kernel, nâng cao BatchNorm/Dropout."""
    return [
        dict(name='baseline', channels=16, kernel_size=3, batchnorm=False, dropout=0.0),
        # TODO: thêm cấu hình, mỗi lần chỉ đổi 1 thuộc tính so với baseline.
    ]

def main():
    set_seed()
    train_ds, val_ds, dataset_cls = get_splits()
    in_channels, side = (1, 28) if DATASET == 'fashion' else (3, 32)
    print('DEVICE:', DEVICE, '| DATASET:', DATASET)
    rows = []
    for config in experiments():
        set_seed()
        model = CNN2Blocks(in_channels, side, config['channels'], config['kernel_size'],
                           config['batchnorm'], config['dropout']).to(DEVICE)
        train_loader, val_loader = make_loaders(train_ds, val_ds)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        print('\\n', config['name'])
        history = []
        for epoch in range(1, EPOCHS+1):
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            history.append((epoch, train_loss, val_loss, val_acc, elapsed))
            print(f'Epoch={epoch}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}, time={elapsed:.2f}s')
        rows.append((config, history))
    # TODO 2: Bảng so sánh, biểu đồ train/validation loss.
    # TODO 3: Chọn model tốt nhất bằng validation accuracy (hòa: validation loss).
    # TODO 4: Tải official test, đánh giá accuracy và confusion matrix (hàng thật, cột đoán).
    # TODO 5: Hiển thị/lưu 5 ảnh dự đoán đúng và 5 ảnh sai (nhãn thật + dự đoán).
    # TODO 6: Nhận xét thay đổi số kênh/kernel và BatchNorm/Dropout (nếu làm).
    print('Hãy hoàn thiện TODO 2–6, xem notebook lời giải để kiểm tra.')

if __name__ == '__main__':
    main()
