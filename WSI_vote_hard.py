import os
from os.path import join as j_
import torch
import torchvision
import numpy as np
from joblib import load
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from model_Msplus import multi_swin_kan_micro_patch4_window7_224 as create_model

from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

class WSIMajorityVoting:
    def __init__(self):
        self.wsi_predictions = defaultdict(list)  # {WSI编号: [Patch预测类别]}
        self.wsi_labels = {}  # {WSI编号: WSI真实类别}
        self.wsi_probs = defaultdict(list)  # {WSI编号: [Patch预测概率]}

    def add_prediction(self, wsi_id, pred_label, true_label, pred_probs=None):
        """
        添加单个 Patch 的预测结果。
        """
        self.wsi_predictions[wsi_id].append(pred_label)
        self.wsi_labels[wsi_id] = true_label
        if pred_probs is not None:
            self.wsi_probs[wsi_id].append(pred_probs)

    def finalize_predictions(self):
        """
        执行硬投票，并生成最终的 WSI 预测结果。
        Returns:
            final_preds: List[int], 所有 WSI 的预测类别
            final_labels: List[int], 所有 WSI 的真实类别
            final_probs: List[np.array], 所有 WSI 的概率分布（可选）
        """
        final_preds = []
        final_labels = []
        final_probs = []

        for wsi_id, patch_preds in self.wsi_predictions.items():
            majority_vote = Counter(patch_preds).most_common(1)[0][0]  # 硬投票
            final_preds.append(majority_vote)
            final_labels.append(self.wsi_labels[wsi_id])

            # 如果有概率信息，计算平均概率
            if wsi_id in self.wsi_probs:
                avg_probs = np.mean(self.wsi_probs[wsi_id], axis=0)
                final_probs.append(avg_probs)
            else:
                final_probs.append(None)

        return final_preds, final_labels, final_probs

    @staticmethod
    def evaluate_predictions(true_labels, pred_labels, class_names, probs=None):
        """
        计算分类指标并输出结果。
        Args:
            true_labels: List[int], 真实类别
            pred_labels: List[int], 预测类别
            class_names: List[str], 类别名称
            probs: List[np.array], 每个样本的概率分布（可选）
        """
        true_labels = np.array(true_labels)
        print(true_labels)
        pred_labels = np.array(pred_labels)
        print(pred_labels)
        
        # 计算指标
        acc = accuracy_score(true_labels, pred_labels)
        bacc = balanced_accuracy_score(true_labels, pred_labels)
        kappa = cohen_kappa_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='macro')
        
        auroc = np.nan
        if probs is not None and len(probs) > 0 and all(p is not None for p in probs):
            try:
                probs = np.array(probs)
                auroc = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"Error calculating AUROC: {e}")

        # 输出结果
        print(f"Acc: {acc:.4f}")
        print(f"BACC: {bacc:.4f}")
        print(f"KAPPA: {kappa:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"AUROC: {auroc:.4f}")
        
        return {
            'Acc': acc,
            'BACC': bacc,
            'KAPPA': kappa,
            'F1': f1,
            'AUROC': auroc
        }


# 初始化设备和模型
device = torch.device("cuda:0")
# 加载模型权重
model_dir = '/home/jiangshuo/CMSwinKAN/TCGARAW-MICRO'
best_model_path = os.path.join(model_dir, 'best_model.pth')

if os.path.exists(best_model_path):
    model = torch.load(best_model_path, map_location='cpu', weights_only=False).to(device)
else:
    raise FileNotFoundError(f"Model file {best_model_path} not found.")

# 确保使用与训练时相同的数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model.eval()

# 数据路径
dataroot = '/home/changshuo/newdataset/TCGA'
# test_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'test'), transform=transform)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        """
        Override __getitem__ to return the path along with the image and label.
        """
        image, label = super().__getitem__(index)
        path = self.samples[index][0]  # Path is stored in self.samples
        return image, label, path

# 使用扩展后的数据集类
test_dataset = ImageFolderWithPaths(
    j_(dataroot, 'test'), 
    transform=transform
)

print("\n类别名称到索引的映射关系：")
print(test_dataset.class_to_idx)  # 这会输出类似 {'class0': 0, 'class1': 1} 的字典
print("类别顺序列表：", test_dataset.classes)  # 显示实际类别名称的顺序

def collate_fn_with_paths(batch):
    """
    自定义的 collate_fn，用来提取路径信息，并返回图像、标签、路径。
    """
    images, labels, paths = zip(*batch)  # 解包，paths 是 batch 中每个 patch 的路径
    
    # 将图像堆叠成一个 batch
    images = torch.stack(images, dim=0)

    # 将标签转为 tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels, list(paths)

# 使用自定义 collate_fn
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_fn_with_paths
)

soft_vote = False

if soft_vote:
    svm_model_path = "model/svm_model_gpu.joblib"
    svm_model = load(svm_model_path)
    majority_voter = WSISoftVoting(svm_model=svm_model)
    print("Soft vote!")
else:
    majority_voter = WSIMajorityVoting()
    print("Hard vote!")


# 测试并聚合 Patch 分类结果
with torch.no_grad():
    for images, labels, paths in tqdm(test_dataloader, desc="Testing Progress"):
        images = images.to(device)
        labels = labels.to(device)
        
        # 预测
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # 根据文件名提取 WSI 编号，并存储结果
        for path, pred, label in zip(paths, predicted.cpu().numpy(), labels.cpu().numpy()):
            filename = os.path.basename(path)
            wsi_id = filename.split('_')[1]  # 提取 WSI 编号
            # print(f"Processing patch from WSI {wsi_id} with true label {label} and predicted label {pred}")  # 日志输出
            majority_voter.add_prediction(wsi_id, pred, label)

# 汇总最终结果并评估
final_preds, final_labels, final_probs = majority_voter.finalize_predictions()
majority_voter.evaluate_predictions(final_labels, final_preds, class_names=test_dataset.classes, probs=final_probs)