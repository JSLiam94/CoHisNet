import os
import sys
import json
import pickle
import random
from metrics import get_eval_metrics,print_metrics
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt


def read_split_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 定义训练集和验证集的路径
    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "test")

    # 检查 train 和 val 文件夹是否存在
    assert os.path.exists(train_root), "train folder does not exist in {}".format(root)
    assert os.path.exists(val_root), "val folder does not exist in {}".format(root)

    # 遍历 train 文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 遍历 train 文件夹
    for cla in flower_class:
        cla_train_path = os.path.join(train_root, cla)
        cla_val_path = os.path.join(val_root, cla)

        # 获取 train 文件夹下的所有图片路径
        train_images = [os.path.join(cla_train_path, i) for i in os.listdir(cla_train_path)
                        if os.path.splitext(i)[-1] in supported]
        train_images.sort()  # 排序，保证各平台顺序一致

        # 获取 val 文件夹下的所有图片路径
        val_images = [os.path.join(cla_val_path, i) for i in os.listdir(cla_val_path)
                      if os.path.splitext(i)[-1] in supported]
        val_images.sort()  # 排序，保证各平台顺序一致

        # 获取该类别对应的索引
        image_class = class_indices[cla]

        # 记录该类别的样本数量
        every_class_num.append(len(train_images) + len(val_images))

        # 将 train 图片路径和标签加入训练集
        for img_path in train_images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

        # 将 val 图片路径和标签加入验证集
        for img_path in val_images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def print_metrics_txt(metrics, file=None):
    if file is None:
        file = sys.stdout  # 默认输出到控制台
    for key, value in metrics.items():
        file.write(f"{key}: {value}\n")
def train_one_epoch(model, optimizer, data_loader, device, epoch, accumulation_steps=4):
    """
    训练一个 epoch 的改进版本。
    支持混合精度训练、梯度累积、学习率调度器等功能。

    参数:
        model: 模型
        optimizer: 优化器
        data_loader: 数据加载器
        device: 设备（如 'cuda' 或 'cpu'）
        epoch: 当前 epoch
        accumulation_steps: 梯度累积步数（默认为 4）

    返回:
        train_loss: 平均训练损失
        train_acc: 平均训练准确率
    """
    

    model.train()  # 设置模型为训练模式
    loss_function = torch.nn.CrossEntropyLoss()  # 定义损失函数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()  # 清空梯度
    scaler = GradScaler()  # 初始化 GradScaler，用于混合精度训练

    sample_num = 0  # 累计样本数
    data_loader = tqdm(data_loader, file=sys.stdout)  # 使用 tqdm 显示进度条

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]  # 更新累计样本数

        # 混合精度训练
        with autocast(device_type=device.type):  # 启用混合精度，指定设备类型
            pred = model(images.to(device))  # 前向传播
            pred_classes = torch.max(pred, dim=1)[1]  # 获取预测类别
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()  # 累计正确预测数

            loss = loss_function(pred, labels.to(device))  # 计算损失
            loss = loss / accumulation_steps  # 梯度累积：损失归一化

        # 反向传播
        scaler.scale(loss).backward()  # 缩放梯度并反向传播

        # 梯度累积：每 accumulation_steps 步更新一次参数
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新 GradScaler
            optimizer.zero_grad()  # 清空梯度

        accu_loss += loss.detach() * accumulation_steps  # 恢复损失值并累计

        # 更新进度条描述
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )

        # 检查损失是否为有限值
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    # 返回平均损失和准确率
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
@torch.no_grad()
def evaluate(model, data_loader, device, epoch,txt):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    all_preds_val = []
    all_targets_val = []
    all_probs_val = []
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        probs = torch.nn.functional.softmax(pred, dim=-1)
        all_probs_val.extend(probs.cpu().numpy())
        all_preds_val.extend(pred_classes.cpu().numpy())  # 将预测值添加到列表中
        all_targets_val.extend(labels.cpu().numpy())
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.4f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    #使用metrics.py
    # 计算验证指标
    val_metrics = get_eval_metrics(all_targets_val, all_preds_val, probs_all=all_probs_val, prefix="val_")
            
    # 打印指标
    print("Validation Metrics:")
    print_metrics(val_metrics)
            
            # 打开一个文件用于写入
    with open(txt, 'a') as file:
                # 写入指标
        file.write('epoch :'+str(epoch)+'\n')        
        file.write("\nValidation Metrics:\n")
        print_metrics_txt(val_metrics, file=file)


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
