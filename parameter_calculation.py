from torchvision.models import alexnet
import torch
from thop import profile
from model_CMs import multi_swin_kan_micro_patch4_window7_224 as create_model
print('multi_swinkan_micro_patch4_window7_224')


# 创建模型实例
#model = alexnet()
#device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=5)
#model = torch.load("./TEST/best_model.pth",weights_only=False)


# input = torch.randn(1, 3, 224, 224)

# # 使用thop库计算参数量和计算量
# flops, params = profile(model, inputs=(input,))
# print(f"FLOPs: {flops}")
# print(f"参数量: {params}")

params = list(model.parameters())
k = 0
for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
print("总参数数量和：" + str(k))
