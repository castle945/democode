"""
LeNet FashionMNIST wandb.sweep
Notes:
    1. 不采用 sdk 上传的方式，传入 sweep_id，启动 agent 执行
    2. 命令行创建 sweep: wandb sweep --project LeNet_FashionMNIST sweep_config_random_search.yaml
       args 传入 sweep_id: python LeNet_FashionMNIST_wandb_sweep_sdk.py --sweep_id city945/LeNet_FashionMNIST/1tz1n8oq
"""
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import wandb

from easydict import EasyDict
cfg = EasyDict()

cfg.batch_size = 256
cfg.epochs = 20
cfg.lr = 0.01

class LeNet(nn.Module):
    def __init__(self, act_layer='Sigmoid'):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Sigmoid() if act_layer == 'Sigmoid' else nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid() if act_layer == 'Sigmoid' else nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120), # nn.Linear(128*4*4, 120),
            nn.Sigmoid() if act_layer == 'Sigmoid' else nn.ReLU(),
            nn.Linear(120, 84),     # nn.Linear(512, 128),
            nn.Sigmoid() if act_layer == 'Sigmoid' else nn.ReLU(),
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
    parser.add_argument('--use_wandb', action='store_true', default=True, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--sweep_id', default=None, help='')
    parser.add_argument('--sweep_num_runs', type=int, default=1, help='')
    
    return parser.parse_args()

def train(args, cfg):
    cfg.batch_size = cfg.batch_size if args.batch_size is None else args.batch_size
    cfg.epochs = cfg.epochs if args.epochs is None else args.epochs

    if args.use_wandb:
        wandb.init(project='LeNet_FashionMNIST', config=cfg, dir='/tmp')
        # wandb.run.tags = ['tag1', 'tag2']
        # wandb.run.name = 'val'
        cfg = wandb.config # for wandb.sweep

    train_dataset = torchvision.datasets.FashionMNIST(root='/datasets/FashionMNIST/train', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    val_dataset = torchvision.datasets.FashionMNIST(root='/datasets/FashionMNIST/test', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size)

    model = LeNet(act_layer=cfg.get('act_layer', 'Sigmoid')).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    accumulated_iter = 0
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

            if args.use_wandb and (accumulated_iter % args.logger_iter_interval == 0):
                # 此数据集太小故在每轮结束额外计算每轮的损失和准确率，一般是每个 iter 记录 iter 的 loss，这里折中，间隔 N 个 iter 记录当前 iter 的 loss
                train_metrics = {
                    "train/train_loss": loss, 
                }
                wandb.log(train_metrics)

            accumulated_iter += 1
        
        train_loss, train_acc = loss_sum / len(train_loader), acc_sum / len(train_loader.dataset)
        val_loss, val_acc = eval(model, val_loader, loss_fn)
        print(f'epoch {epoch+1}, train loss {train_loss:.3f}, train acc {train_acc:.3f}, val loss {val_loss:.3f}, val acc {val_acc:.3f}')
        if args.use_wandb:
            val_metrics = {
                "eval/epoch": epoch,
                "eval/val_loss": val_loss,
                "eval/val_acc": val_acc,
                "eval/train_loss": train_loss,
                "eval/train_acc": train_acc,
            }
            wandb.log(val_metrics)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':

    args = parse_args()

    if args.sweep_id is not None:
        from functools import partial
        # count: 本次代理执行多少次搜索，每次搜索会从 wandb 获取参数
        wandb.agent(args.sweep_id, partial(train, args, cfg), count=args.sweep_num_runs)
    else:
        train(args, cfg)
