import torch

from torchvision import datasets, transforms, models
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from torch.utils.data import DataLoader
from tqdm import tqdm

batch_size = 256
max_epochs = 40
lr = 0.008
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval(eval_loader, model):
    acc_sum, n = 0.0, 0
    for input, label in eval_loader:
        model.eval()
        input, label = input.to(device), label.to(device)
        pred = model(input)
        acc_sum += (pred.argmax(dim=1) == label).sum().cpu().item()
        model.train()
        n += label.shape[0]

    return acc_sum / n

def main():
    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]),
        "val": transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ])
    }

    train_dataset = datasets.CIFAR10('/datasets/CIFAR10', train=True, transform=data_transform['train'], download=True)
    val_dataset = datasets.CIFAR10('/datasets/CIFAR10', train=False, transform=data_transform['val'], download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = models.resnet18().to(device)
    optim = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    for epoch in range(max_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for input, label in tqdm(train_loader):
            input, label = input.to(device), label.to(device)
            pred = model(input)
            loss = loss_fn(pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += (pred.argmax(dim=1) == label).sum().cpu().item()
            n += label.shape[0]
        
        train_acc = train_acc_sum / n
        test_acc = eval(val_loader, model)
        print(f'epoch {epoch+1}, loss sum {train_loss_sum:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

if __name__ == '__main__':
    main()

"""
CUDA_VISIBLE_DEVICES=1 python3 train_pytorch.py
"""