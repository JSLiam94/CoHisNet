from PIL import Image
import torch
from torch.utils.data import Dataset
from PIL import UnidentifiedImageError

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)



    def __getitem__(self, item):
        try:
            img = Image.open(self.images_path[item])
            
            # 如果图像不是 RGB 模式，转换为 RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            label = self.images_class[item]

            if self.transform is not None:
                img = self.transform(img)

            return img, label
        except (UnidentifiedImageError, OSError) as e:
            print(f"Error loading image {self.images_path[item]}: {e}")
            # 返回一个占位符图像和标签
            placeholder_img = Image.new('RGB', (224, 224), color=(0, 0, 0))  # 黑色图像
            placeholder_label = 0  # 默认标签
            if self.transform is not None:
                placeholder_img = self.transform(placeholder_img)
            return placeholder_img, placeholder_label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
