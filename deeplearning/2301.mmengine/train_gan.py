"""
参考: https://mmengine.readthedocs.io/zh_CN/latest/examples/train_a_gan.html
"""
import numpy as np
from mmcv.transforms import to_tensor
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from mmengine.dataset import BaseDataset

class MNISTDataset(BaseDataset):
    def __init__(self, data_root, pipeline, test_mode=False):
        if test_mode:
            # ? 怎么 test_mode=True 是训练集
            # MNIST 包含 60000 个样本的训练集，10000 个样本的测试集，取的是训练集中的 55000 个样本
            mnist_full = MNIST(data_root, train=True, download=True)
            self.mnist_dataset, _ = random_split(mnist_full, [55000, 5000])
        else:
            # 取的是 10000 个样本的测试集
            self.mnist_dataset = MNIST(data_root, train=False, download=True)

        super().__init__(
            data_root=data_root, pipeline=pipeline, test_mode=test_mode)

    @staticmethod
    def totensor(img):
        # (28, 28) -> (28, 28, 1) -> (1, 28, 28)
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return to_tensor(img)

    def load_data_list(self):
        return [
            # （PIL.Image, label) 的元组，取 x[0] 得到 (28, 28) 的图片
            dict(inputs=self.totensor(np.array(x[0]))) for x in self.mnist_dataset
        ]

dataset = MNISTDataset("/datasets/MNIST/train", [])



import os
import torch
from mmengine.runner import Runner

NUM_WORKERS = int(os.cpu_count() / 2)
BATCH_SIZE = 256

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset)
train_dataloader = Runner.build_dataloader(train_dataloader)



import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.noise_size = noise_size

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 生成器就是一堆堆叠的全连接层，最后展平成图片
        # noise_size 就是第一层的输入通道
        self.model = nn.Sequential(
            *block(noise_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            # np.prod 计算所有元素的乘积，即 28 * 28
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        # 判别器也是一堆堆叠的全连接层，就是把展平的图片送进全连接层得到一个二分类结果
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

generator = Generator(100, (1, 28, 28))
discriminator = Discriminator((1, 28, 28))



from mmengine.model import ImgDataPreprocessor
# 数据归一化
data_preprocessor = ImgDataPreprocessor(mean=([127.5]), std=([127.5]))

import torch.nn.functional as F
from mmengine.model import BaseModel

# 锁定训练生成器时判别器的权重
def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not.
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class GAN(BaseModel):
    def __init__(self, generator, discriminator, noise_size,
                 data_preprocessor):
        super().__init__(data_preprocessor=data_preprocessor)
        assert generator.noise_size == noise_size
        self.generator = generator
        self.discriminator = discriminator
        self.noise_size = noise_size

    def train_step(self, data, optim_wrapper):
        # 获取数据和数据预处理
        inputs_dict = self.data_preprocessor(data, True)
        # 训练判别器
        disc_optimizer_wrapper = optim_wrapper['discriminator']
        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(inputs_dict,
                                                disc_optimizer_wrapper)

        # 训练生成器
        set_requires_grad(self.discriminator, False)
        gen_optimizer_wrapper = optim_wrapper['generator']
        with gen_optimizer_wrapper.optim_context(self.generator):
            log_vars_gen = self.train_generator(inputs_dict,
                                                gen_optimizer_wrapper)

        set_requires_grad(self.discriminator, True)
        log_vars.update(log_vars_gen)

        return log_vars

    def forward(self, batch_inputs, data_samples=None, mode=None):
        return self.generator(batch_inputs)

    def disc_loss(self, disc_pred_fake, disc_pred_real):
        # 计算判别器的损失，判别器应该使生成图片的预测结果尽可能为0，原始图片的预测结果尽可能为1
        losses_dict = dict()
        # 预测值与目标值越接近，损失越小
        losses_dict['loss_disc_fake'] = F.binary_cross_entropy(
            disc_pred_fake, 0. * torch.ones_like(disc_pred_fake))
        losses_dict['loss_disc_real'] = F.binary_cross_entropy(
            disc_pred_real, 1. * torch.ones_like(disc_pred_real))

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake):
        # 计算生成器的损失，生成器应该使生成图片的预测结果尽可能为1
        losses_dict = dict()
        losses_dict['loss_gen'] = F.binary_cross_entropy(
            disc_pred_fake, 1. * torch.ones_like(disc_pred_fake))
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def train_discriminator(self, inputs, optimizer_wrapper):
        real_imgs = inputs['inputs']
        z = torch.randn(
            (real_imgs.shape[0], self.noise_size)).type_as(real_imgs)
        with torch.no_grad():
            fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real)
        optimizer_wrapper.update_params(parsed_losses)
        return log_vars

    def train_generator(self, inputs, optimizer_wrapper):
        real_imgs = inputs['inputs']
        z = torch.randn(real_imgs.shape[0], self.noise_size).type_as(real_imgs)

        fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars


model = GAN(generator, discriminator, 100, data_preprocessor)



from mmengine.optim import OptimWrapper, OptimWrapperDict

opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_g_wrapper = OptimWrapper(opt_g)

opt_d = torch.optim.Adam(
    discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_d_wrapper = OptimWrapper(opt_d)

opt_wrapper_dict = OptimWrapperDict(
    generator=opt_g_wrapper, discriminator=opt_d_wrapper)



train_cfg = dict(by_epoch=True, max_epochs=220)
runner = Runner(
    model,
    work_dir='work_dirs/',
    train_dataloader=train_dataloader,
    train_cfg=train_cfg,
    optim_wrapper=opt_wrapper_dict)
runner.train()

# z = torch.randn(64, 100).cuda()
# img = model(z)

# from torchvision.utils import save_image
# save_image(img, "result.png", normalize=True)