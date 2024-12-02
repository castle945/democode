import torch

from model import LeNet
from torchvision.datasets import mnist
from torchvision.datasets import FashionMNIST
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

batch_size = 256
max_epochs = 20
lr = 0.01
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
    # train_dataset = mnist.MNIST(root='/datasets/MNIST/train', train=True, transform=ToTensor())
    # test_dataset = mnist.MNIST(root='/datasets/MNIST/test', train=False, transform=ToTensor())
    train_dataset = FashionMNIST(root='/datasets/FashionMNIST/train', train=True, transform=ToTensor(), download=True)
    test_dataset = FashionMNIST(root='/datasets/FashionMNIST/test', train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LeNet().to(device)
    # optim = SGD(model.parameters(), lr=lr) # SGD 不收敛
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
        test_acc = eval(test_loader, model)
        print(f'epoch {epoch+1}, loss sum {train_loss_sum:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

if __name__ == '__main__':
    main()

# MNIST 下载有问题则直接从其他地方下，https://github.com/ChawDoe/LeNet5-MNIST-PyTorch -> ./train/MNIST/raw