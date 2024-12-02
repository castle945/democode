"""
LeNet FashionMNIST wandb.sweep
Notes:
    1. 上传配置文件到 wandb 控制器: wandb sweep --project <propject-name> <path-to-config file>
       本地启动代理，获取超参执行 wandb agent <sweep-ID>
    2. wandb agent 会将接收到的超参以命令行参数的形式传入，与 argparse 冲突，故需要将 sweep 的超参加到 argparse 或者使用 parse_known_args，此外希望通过命令行修改的参数需要写在 yaml 文件的 command 配置中
    3. sweep_config 都要加上 metric 才能看到每组超参数的效果，其字段需与 wandb.log 记录的名称相同
    4. 存在连续值的网格搜索、随机搜索、贝叶斯搜索将永远运行，直到手动杀死，故可以 wandb agent --count 5 指定搜索次数
    5. 多卡同时搜索，CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
    6. 暂停/恢复搜索: 
        wandb sweep --pause entity/project/sweep_ID
        wandb sweep --resume entity/project/sweep_ID
    7. 重新搜索: 搜索到一半时程序崩溃，wandb 界面上 resume 使用新 id 重新搜索，已搜索参数不会重新执行 https://docs.wandb.ai/guides/sweeps/faq
    8. hyperband 算法: 预设迭代次数，到达迭代次数时比较指标是否过高或过低，停止运行
        参数: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#hyperband
        预设迭代次数列表为:
            给定 min_iter, eta: [min_iter, min_iter*eta, min_iter*(eta)**2, min_iter*(eta)**3, ...]
            给定 max_iter, eta, s: [max_iter/(eta)**s, max_iter/(eta)**(s-1)...len=s] 
            注意此 iter 为 wandb.log 记录的频率，需要结合 logger_iter_interval 确定
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
    # parser.add_argument('--act_layer', default=None, help='')
    # parser.add_argument('--lr', default=None, help='')
    
    # return parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args

if __name__ == '__main__':

    args = parse_args()
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
