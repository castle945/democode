import torch

from model import *
from torchvision import datasets, transforms, models
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from torch.utils.data import DataLoader
from tqdm import tqdm
import os

batch_size = 16
max_epochs = 20
lr = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_root = "/datasets/flower5/"
# data_root = "/datasets/imagenet5/"

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
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]),
        "val": transforms.Compose([ transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])}

    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=data_transform['train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=data_transform['val'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # model = AlexNet(num_classes=5, init_weights=True).to(device)
    # model = VGG16(num_classes=5).to(device)
    # optim = Adam(model.parameters(), lr=lr)
    
    model = models.vgg16(pretrained=True).to(device)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 5).to(device)
    # 只微调全连接层
    # optim = Adam(model.classifier[-1].parameters(), lr=lr)
    # 微调所有层
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
CUDA_VISIBLE_DEVICES=1 python3 train.py
"""