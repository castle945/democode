"""
LeNet FashionMNIST baseline
Notes:
    1. FashionMNIST ，10 类，衬衫、鞋等
"""
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn

from easydict import EasyDict
cfg = EasyDict()

cfg.batch_size = 256
cfg.epochs = 20
cfg.lr = 0.01

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Sigmoid(),   # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),   # nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120), # nn.Linear(128*4*4, 120),
            nn.Sigmoid(),   # nn.ReLU(),
            nn.Linear(120, 84),     # nn.Linear(512, 128),
            nn.Sigmoid(),   # nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        feature = self.conv(x)  # torch.Size([8, 16, 4, 4])
        output = self.fc(feature.view(x.shape[0], -1))
        return output

def eval(model, val_loader, loss_fn):
    loss_sum, acc_sum = 0.0, 0.0
    model.eval()
    for input, label in val_loader:
        input, label = input.cuda(), label.cuda()
        pred = model(input)
        loss = loss_fn(pred, label)

        loss_sum += loss.cpu().item()
        acc_sum += (pred.argmax(dim=1) == label).sum().cpu().item()

    model.train()
    return loss_sum / len(val_loader), acc_sum / len(val_loader.dataset)

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', default=None, help='batch size in training')
    parser.add_argument('--epochs', default=None, help='number of epoch in training')
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    cfg.batch_size = cfg.batch_size if args.batch_size is None else args.batch_size
    cfg.epochs = cfg.epochs if args.epochs is None else args.epochs

    train_dataset = torchvision.datasets.FashionMNIST(root='/datasets/FashionMNIST/train', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    val_dataset = torchvision.datasets.FashionMNIST(root='/datasets/FashionMNIST/test', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size)

    model = LeNet().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        loss_sum, acc_sum = 0.0, 0.0
        for input, label in tqdm(train_loader):
            input, label = input.cuda(), label.cuda()
            pred = model(input)

            loss = loss_fn(pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_sum += loss.cpu().item()
            acc_sum += (pred.argmax(dim=1) == label).sum().cpu().item()
        
        train_loss, train_acc = loss_sum / len(train_loader), acc_sum / len(train_loader.dataset)
        val_loss, val_acc = eval(model, val_loader, loss_fn)
        print(f'epoch {epoch+1}, train loss {train_loss:.3f}, train acc {train_acc:.3f}, val loss {val_loss:.3f}, val acc {val_acc:.3f}')
