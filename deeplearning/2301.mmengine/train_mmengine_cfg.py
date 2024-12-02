"""
参考: https://mmengine.readthedocs.io/zh_CN/latest/get_started/15_minutes.html
注册 CIFAR10 数据集、注册模型和评价指标
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

# $$ 注册模型和评价指标
from mmengine.registry import MODELS, METRICS
@MODELS.register_module('MMResNet')
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

@METRICS.register_module('Accuracy')
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

# $$ 注册 CIFAR10 数据集
from mmengine.registry import DATASETS, TRANSFORMS 
from mmengine.dataset.base_dataset import Compose
import torchvision.transforms as tvt
# 注册 torchvision 中用到的数据预处理模块
# mmcv 中也有注册 ToTensor，且参数不一样
TRANSFORMS.register_module('RandomCrop', module=tvt.RandomCrop)
TRANSFORMS.register_module('RandomHorizontalFlip', module=tvt.RandomHorizontalFlip)
TRANSFORMS.register_module('MyToTensor', module=tvt.ToTensor)
TRANSFORMS.register_module('MyNormalize', module=tvt.Normalize)
# 注册 torchvision 的 CIFAR10 数据集
# 数据预处理也需要在此一起构建
@DATASETS.register_module(name='Cifar10', force=False)
def build_torchvision_cifar10(transform=None, **kwargs):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = Compose(transform)
    return datasets.CIFAR10(**kwargs, transform=transform)

def main():
    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_dataloader = dict(
        batch_size=batch_size,
        dataset=dict(
            type='Cifar10',
            root='/datasets/CIFAR10',
            train=True,
            transform=[
                dict(type='RandomCrop', size=32, padding=4),
                dict(type='RandomHorizontalFlip'),
                dict(type='MyToTensor'),
                dict(type='MyNormalize', **norm_cfg)
            ],
            download=True
        ),
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate')
    )
    val_dataloader = dict(
        batch_size=batch_size,
        dataset=dict(
            type='Cifar10',
            root='/datasets/CIFAR10',
            train=False,
            transform=[
                dict(type='MyToTensor'),
                dict(type='MyNormalize', **norm_cfg)
            ],
            download=True
        ),
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate')
    )

    runner = Runner(
        # model=MMResNet(),
        model=dict(type='MMResNet'), # 注册模型
        work_dir='work_dirs',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type=Adam, lr=lr)),
        train_cfg=dict(by_epoch=True, max_epochs=max_epochs, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        # val_evaluator=dict(type=Accuracy),
        val_evaluator=dict(type='Accuracy'), # 注册评价指标
    )
    runner.train()

if __name__ == '__main__':
    main()

# # $$ 从配置文件训练
# from mmengine.config import Config
# from mmengine.runner import Runner
# config = Config.fromfile('example_config.py')
# runner = Runner.from_cfg(config)
# runner.train()