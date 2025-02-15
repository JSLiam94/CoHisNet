import os
import argparse
from torchvision import datasets, transforms
import torch
from torch.nn.utils import prune
import torch.optim as optim
from torch.utils.data import DataLoader
#from my_dataset import MyDataSet
#from model import swinkan_base_patch4_window7_224 as create_model
from model_CMs import multi_swin_kan_micro_patch4_window7_224 as create_model
print('multi_swin_kan_micro_patch4_window7_224')
from utils import read_split_data, train_one_epoch, evaluate,evaluate_confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 引入学习率调度器

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    w = 224
    h = 224
    data_transforms = transforms.Compose([
        transforms.Resize((w, h)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if os.path.exists("./multi_swin_kan_micro_patch4_window7_224") is False:
        os.makedirs("./multi_swin_kan_micro_patch4_window7_224")

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 实例化训练数据集
    DATA_DIR = args.data_path
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    val_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
    # 打印数据信息
    print("len of train dataset:" + str(len(train_dataset)))
    print("len of val dataset:" + str(len(val_dataset)))
    print(train_dataset.classes)
    print(train_dataset.class_to_idx)

    num_workers = 8
    print('Using {} dataloader workers every process'.format(num_workers))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print("Data has been loaded")
    print("Loading model")
    #model = torch.load("/home/zhushenghao/data/JS/sk/multi_swin_kan_micro_patch4_window7_224/model.pth",weights_only=False)
    

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        model = torch.load(args.weights, map_location=device, weights_only=False)
        #model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    else:
            model = create_model(num_classes=args.num_classes, device=device).to(device)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):  # 对卷积层进行剪枝
                    prune.l1_unstructured(module, name='weight', amount=0.2)  # 剪枝 20%
            print("Model has been constructed")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-3)

    # 动态调整学习率：使用 ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

    # 早停机制参数
    early_stopping_patience = 5  # 如果验证集损失连续 5 次没有提升，则停止训练
    early_stopping_counter = 0
    best_val_loss = float('inf')  # 初始化最佳验证集损失

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )
        torch.save(model, "./multi_swin_kan_micro_patch4_window7_224/model-{}.pth".format(epoch))

        # 验证
        if (epoch + 1) % val_epoch == 0:
            val_loss, val_acc = evaluate_confusion_matrix(
                model=model,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
                txt="./multi_swin_kan_micro_patch4_window7_224/val.txt"
            )

            # 动态调整学习率
            scheduler.step(val_loss)  # 根据验证集损失调整学习率

            # 将结果写入日志文件
            with open("./log.txt", "a") as f:
                f.write(f"epoch:{epoch} train_loss:{train_loss} train_acc:{train_acc} val_loss:{val_loss} val_acc:{val_acc}\n")


            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0  # 重置计数器
                # 保存最佳模型
                torch.save(model, "./multi_swin_kan_micro_patch4_window7_224/best_model.pth")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}!")
                    with open("./log-fu.txt", "a") as f:
                        f.write(f"Early stopping triggered at epoch {epoch}!")

                    break  # 停止训练

            
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="/home/zhengjingyuan/data/ZDRN1.2")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--start_epoch', type=int, default=0, help='training-start-epoch')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    val_epoch = 2  # 每隔 5 个 epoch 验证一次
    main(opt)
