"""
LeNet FashionMNIST wandb.table
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

def eval(model, val_loader, loss_fn, epoch, use_wandb=False, table=None, logger_epoch_interval=5, num_batchs_to_log=10, num_sample_to_log_per_batch=16):
    """
    Args:
        logger_epoch_interval: 每隔多少个 epoch 记录一次
        num_batchs_to_log: 每次记录多少个 batch 的数据
        num_sample_to_log_per_batch: 每个 batch 记录多少个样本
    Notes:
        1. 以验证期间为例，训练时也可以，不一定要展示图片，只展示预测标签、真值、IoU 等等都可以
        2. id 为 epoch + sample_id + batch_id ，设置 row["id"].split("_")[-1] == "20" 来查看某个样本的所以预测情况
        3. 可以设置 row["pred"] != row["label"] 来过滤查看错误样本
    """
    use_wandb_table = use_wandb and table and (epoch % logger_epoch_interval == 0)
    batch_id = 0
    loss_sum, acc_sum = 0.0, 0.0
    model.eval()
    for input, label in val_loader:
        input, label = input.cuda(), label.cuda()
        pred = model(input)
        loss = loss_fn(pred, label)

        loss_sum += loss.cpu().item()
        acc_sum += (pred.argmax(dim=1) == label).sum().cpu().item()

        if use_wandb_table and (batch_id < num_batchs_to_log):
            pred_label = pred.argmax(dim=1)
            # log_scores = torch.functional.F.softmax(pred.data, dim=1)
            n = num_sample_to_log_per_batch
            log_images, log_labels, log_preds = input[:n].detach().cpu().numpy(), label[:n].detach().cpu().numpy(), pred_label[:n].detach().cpu().numpy()
            sample_id = 0
            for log_image, log_label, log_pred in zip(log_images, log_labels, log_preds):
                id = f"{epoch}_{batch_id}_{sample_id}"
                table.add_data(id, wandb.Image(log_image), log_pred, log_label)
                sample_id += 1
            batch_id += 1

    model.train()
    return loss_sum / len(val_loader), acc_sum / len(val_loader.dataset)

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', default=None, help='batch size in training')
    parser.add_argument('--epochs', default=None, help='number of epoch in training')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--watch_model', action='store_true', default=False, help='')
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    cfg.batch_size = cfg.batch_size if args.batch_size is None else args.batch_size
    cfg.epochs = cfg.epochs if args.epochs is None else args.epochs

    if args.use_wandb:
        wandb.init(project='LeNet_FashionMNIST', config=cfg, dir='/tmp')
        # wandb.run.tags = ['tag1', 'tag2']
        # wandb.run.name = 'val'

        columns = ["id", "image", "pred", "label"]
        table = wandb.Table(columns=columns)

    train_dataset = torchvision.datasets.FashionMNIST(root='/datasets/FashionMNIST/train', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    val_dataset = torchvision.datasets.FashionMNIST(root='/datasets/FashionMNIST/test', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    # train_dataset = torchvision.datasets.MNIST(root='/datasets/MNIST/train', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    # val_dataset = torchvision.datasets.MNIST(root='/datasets/MNIST/test', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size)

    model = LeNet().cuda()
    if args.watch_model:
        wandb.watch(model, log='all', log_graph=True)
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
        val_loss, val_acc = eval(model, val_loader, loss_fn, epoch, use_wandb=args.use_wandb, table=table)
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
        wandb.log({"preds_table": table})
        wandb.finish()
