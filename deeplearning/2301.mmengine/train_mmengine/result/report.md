##### 实验目的
比较使用原生 Pytorch 训练与使用 mmengine 训练，训练速度等表现

##### 实验结论
使用 `mmengine` 训练速度慢了一倍多，精度差不多；


##### 实验结果
```
sh -c 'date && CUDA_VISIBLE_DEVICES=1 python3 train_mmengine.py && date' >> train_mmengine.log
sh -c 'date && CUDA_VISIBLE_DEVICES=1 python3 train_pytorch.py && date' >> train_pytorch.log 
```
[train_mmengine.log](train_mmengine.log) 
```
2023年 02月 08日 星期三 16:34:23 CST
2023年 02月 08日 星期三 16:40:05 CST

```
[train_pytorch.log](train_pytorch.log)
```
2023年 02月 08日 星期三 16:29:57 CST
2023年 02月 08日 星期三 16:32:08 CST

```