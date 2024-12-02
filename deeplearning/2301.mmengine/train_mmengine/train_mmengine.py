"""
参考: https://mmengine.readthedocs.io/zh_CN/latest/get_started/15_minutes.html
mmengine Cifar10 Resnet
"""
from torchvision import datasets, transforms, models
from mmengine.runner import Runner
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric

import torch.nn.functional as F
from torch.optim import Adam

batch_size = 256
max_epochs = 40
lr = 0.008

class MMResNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18()
    
    def forward(self, inputs, labels, mode):
        x = self.resnet(inputs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu()
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)

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

    train_dataloader = dict(
        batch_size=batch_size,
        dataset=train_dataset,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate')
    )
    val_dataloader = dict(
        batch_size=batch_size,
        dataset=val_dataset,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate')
    )

    runner = Runner(
        model=MMResNet(),
        work_dir='work_dirs',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type=Adam, lr=lr)),
        train_cfg=dict(by_epoch=True, max_epochs=max_epochs, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
    )
    runner.train()

if __name__ == '__main__':
    main()