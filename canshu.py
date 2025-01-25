from torchvision.models import alexnet
import torch
from thop import profile
import os
import argparse
from torchvision import datasets, transforms
import torch
from torch.nn.utils import prune
import torch.optim as optim
from torch.utils.data import DataLoader
from my_dataset import MyDataSet
#from model import swinkan_base_patch4_window7_224 as create_model
from model import swin_base_patch4_window7_224 as create_model
print('swin_base_patch4_window7_224')
from utils import read_split_data, train_one_epoch, evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 引入学习率调度器

# 创建模型实例
#model = alexnet()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

model = create_model(num_classes=10)
input = torch.randn(1, 3, 224, 224)

# 使用thop库计算参数量和计算量
flops, params = profile(model, inputs=(input,))
print(f"FLOPs: {flops}")
print(f"参数量: {params}")