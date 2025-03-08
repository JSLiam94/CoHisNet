import os
from os.path import join as j_
import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from uni import get_encoder
from scipy.special import softmax
from joblib import load
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from model_Msplus import multi_swin_kan_micro_patch4_window7_224 as create_model

from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class WSIMajorityVoting:
    def __init__(self):
        self.wsi_predictions = defaultdict(list)  # {WSI编号: [Patch预测类别]}
        self.wsi_labels = {}  # {WSI编号: WSI真实类别}

    def add_prediction(self, wsi_id, pred_label, true_label):
        """
        添加单个 Patch 的预测结果。
        """
        self.wsi_predictions[wsi_id].append(pred_label)
        self.wsi_labels[wsi_id] = true_label

    def finalize_predictions(self):
        """
        执行硬投票，并生成最终的 WSI 预测结果。
        Returns:
            final_preds: List[int], 所有 WSI 的预测类别
            final_labels: List[int], 所有 WSI 的真实类别
        """
        final_preds = []
        final_labels = []
        for wsi_id, patch_preds in self.wsi_predictions.items():
            majority_vote = Counter(patch_preds).most_common(1)[0][0]  # 硬投票
            final_preds.append(majority_vote)
            final_labels.append(self.wsi_labels[wsi_id])
        return final_preds, final_labels

    @staticmethod
    def evaluate_predictions(true_labels, pred_labels, class_names):
        """
        计算分类指标并输出结果。
        Args:
            true_labels: List[int], 真实类别
            pred_labels: List[int], 预测类别
            class_names: List[str], 类别名称
        """
        print(true_labels)
        print("----")
        print(pred_labels)
        print("----")
        print(class_names)
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, target_names=class_names)
        print(f"WSI Classification Accuracy: {accuracy:.4f}")
        print("Detailed Classification Report:")
        print(report)
        return accuracy, report


class WSISoftVoting:
    def __init__(self, svm_model, scaler, alpha=10, beta=10, gamma=0.1):
        """
        初始化软投票类。
        Args:
            svm_model: 已加载的 SVM 模型，用于组织成分分类。
            alpha: 神经毡 (neuropil) 的加权系数。
            beta: 施旺基质 (Schwannian stroma) 的加权系数。
            gamma: 基准权重，用于非关键组织成分的权重分配。
        """
        self.svm_model = svm_model
        self.scaler = scaler
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.wsi_predictions = defaultdict(list)  # {WSI编号: [(patch概率分布, 投票权重)]}
        self.wsi_labels = {}  # {WSI编号: WSI真实类别}

    def add_prediction(self, wsi_id, patch_probs, patch_weight, true_label):
        """
        添加单个图像的预测结果和权重。
        Args:
            wsi_id: WSI 编号。
            patch_probs: UNI 模型的预测概率分布。
            patch_weight: SVM 计算出的权重。
            true_label: 真实的 WSI 类别。
        """
        self.wsi_predictions[wsi_id].append((patch_probs, patch_weight))
        self.wsi_labels[wsi_id] = true_label

    def svm_predict(self, features):
        """
        使用 SVM 模型预测组织成分的概率分布。
        Args:
            features: Patch 的输入特征。
        Returns:
            svm_probs: SVM 模型的预测概率分布。
        """
        logits = self.svm_model.predict_proba(features)
        # print("[Debug] SVM logits 形状:", logits.shape)  # 应为 (1, n_classes)
        svm_probs = softmax(logits, axis=1)
        # print("[Debug] SVM 概率分布:", svm_probs)
        return svm_probs[0]

    def calculate_patch_weight(self, svm_probs):
        """
        根据 SVM 的预测结果计算 Patch 的权重。
        Args:
            svm_probs: SVM 的预测概率分布。
        Returns:
            weight: Patch 的权重。
        """
        P_1 = svm_probs[1]  # 神经毡的概率
        P_2 = svm_probs[2]  # 施旺基质的概率
        weight = self.alpha * P_1 + self.beta * P_2
        # 如果该 Patch 不包含关键组织，赋予较低的基准权重
        if np.argmax(svm_probs) == 0:  # 非关键组织
            weight = self.gamma
        return weight

    def preprocess_for_svm(self, image):
        """
        专为 SVM 预处理：调整大小、展平、标准化。
        Args:
            image: PyTorch 张量，形状为 [C, H, W]
        Returns:
            svm_features: 标准化后的展平特征向量（形状 [12288]）
        """
        # 转换为 NumPy 并调整通道顺序
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # HxWxC
        # 调整大小为 64x64
        image_resized = cv2.resize(image_np, (64, 64))
        # 展平并标准化
        image_flattened = image_resized.flatten()
        image_normalized = self.scaler.transform([image_flattened])
        return image_normalized

    def preprocess_for_uni(self, image):
        """
        专为 UNI 模型预处理：直接使用原始图像（或调整到模型需要的尺寸）。
        Args:
            image: PyTorch 张量，形状为 [C, H, W]
        Returns:
            uni_input: 直接返回原始图像（或调整到 UNI 模型需要的尺寸）
        """
        # 如果 UNI 模型需要特定尺寸（如 224x224），在此调整大小
        # 示例：调整到 224x224
        image_resized = torchvision.transforms.functional.resize(image, (224, 224))
        return image_resized

    def finalize_predictions(self):
        """
        执行软投票，生成最终的 WSI 分类结果。
        Returns:
            final_preds: List[int], WSI 的预测类别。
            final_labels: List[int], WSI 的真实类别。
        """
        final_preds = []
        final_labels = []
        all_probs = []  # 收集所有WSI的概率分布
        for wsi_id, patch_info in self.wsi_predictions.items():
            total_weighted_probs = np.zeros(len(patch_info[0][0]))
            total_weight = 0
            for patch_probs, weight in patch_info:
                total_weighted_probs += patch_probs * weight
                total_weight += weight
            wsi_probs = total_weighted_probs / total_weight
            wsi_label = np.argmax(wsi_probs)
            final_preds.append(wsi_label)
            final_labels.append(self.wsi_labels[wsi_id])
            all_probs.append(wsi_probs)
        return final_preds, final_labels, np.array(all_probs)  # 返回概率数组

    @staticmethod
    def evaluate_predictions(true_labels, pred_labels, class_names, probs=None):
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        
        # 计算指标
        acc = accuracy_score(true_labels, pred_labels)
        bacc = balanced_accuracy_score(true_labels, pred_labels)
        kappa = cohen_kappa_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='macro')
        
        auroc = np.nan
        if probs is not None and probs.size > 0:
            try:
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


device = torch.device("cuda:0")

# 加载模型权重
model_dir = '/home/jiangshuo/CMSwinKAN/BCNB-MICRO'
best_model_path = os.path.join(model_dir, 'best_model.pth')

if os.path.exists(best_model_path):
    model = torch.load(best_model_path, map_location='cpu', weights_only=False).to(device)
else:
    raise FileNotFoundError(f"Model file {best_model_path} not found.")

# 确保使用与训练时相同的数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model.eval()

# 数据路径
dataroot = '/home/jiangshuo/data/BCNB'
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
    transform=data_transforms
)

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

soft_vote = True

if soft_vote:
    svm_model, loaded_encoder, scaler = load('./svm_model/svm_model_gpu.joblib')
    majority_voter = WSISoftVoting(svm_model=svm_model, scaler=scaler)
    print("Soft vote!")
else:
    majority_voter = WSIMajorityVoting()
    print("Hard vote!")


with torch.no_grad():
    for images, labels, paths in tqdm(test_dataloader, desc="Testing Progress"):
        images = images.to(device)
        labels = labels.to(device)
        
        # UNI 模型批量预测
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        # 遍历每个图像
        for i in range(images.size(0)):
            image = images[i]          # 形状 [C, H, W]
            label = labels[i].item()   # 真实标签
            path = paths[i]            # 图像路径
            wsi_id = os.path.basename(path).split('_')[0]

            # ----------------------------
            # 预处理和权重计算（在外部完成）
            # ----------------------------
            # 1. 预处理图像供 SVM 使用
            svm_features = majority_voter.preprocess_for_svm(image)  # 形状 [1, 12288]
            # 2. SVM 预测概率
            svm_probs = majority_voter.svm_predict(svm_features)
            # 3. 计算权重
            patch_weight = majority_voter.calculate_patch_weight(svm_probs)

            # 获取 UNI 模型的预测概率
            patch_probs = probabilities[i].cpu().numpy()

            # 添加预测结果（传入所有必需参数）
            majority_voter.add_prediction(
                wsi_id=wsi_id,
                patch_probs=patch_probs,
                patch_weight=patch_weight,
                true_label=label
            )

# 汇总最终结果并评估
final_preds, final_labels, final_probs = majority_voter.finalize_predictions()
WSISoftVoting.evaluate_predictions(final_labels, final_preds, 
                                  class_names=test_dataset.classes, 
                                  probs=final_probs)
