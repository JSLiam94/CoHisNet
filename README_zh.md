CMSwinKAN
=========
**``Read this in other languages: ``[English](README.md)|中文.**
---------
### Contrast Multi-Scale Adaptive Swin KANsformer
### 对比驱动下的多尺度特征融合自适应Swin KANsformer

![image](https://github.com/user-attachments/assets/e374c837-91f7-4d56-9799-2116da9523e7)

## 环境配置

    conda create -n CMSwinKAN python=3.8.20
    conda activate CMSwinKAN
    pip install -r requirements.txt

## 数据集组织
以下是默认的数据集组织方式<br>

    dataset  
     ├── train
     │   ├── class1 
     │   ├── class2  
     │   └── ... 
     └── test
         ├── class1
         ├── class2
         └── ...
## Patch级别（图像级别）训练
 在`train_new.py`中修改`data-path`和 设置`num_classes`、`num_workers`(对于Windows系统, 建议设置为`0`)<br>

    python train_new.py

## Patch级别（图像级别）测试/评估

    python val.py
  
## 引用
### 部分代码引用自：

[Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

[efficient-kan](https://github.com/Blealtan/efficient-kan)

[ConDSeg](https://github.com/Mengqi-Lei/ConDSeg)

[SCKansformer](https://github.com/JustlfC03/SCKansformer)
