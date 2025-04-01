CMSwinKAN
=========
**``Read this in other languages: ``[English](README.md)|中文.**
---------
#### Towards Accurate and Interpretable Neuroblastoma Diagnosis via Contrastive Multi-scale Pathological Image Analysis
#### 通过对比驱动的多尺度病理图像分析进行准确且可解释的神经母细胞瘤诊断
### Contrast Multi-Scale Adaptive Swin KANsformer
### 对比驱动下的多尺度特征融合自适应Swin KANsformer

## 图像级别分类
![CMSwinKAN](https://github.com/user-attachments/assets/a3a72e0a-7df6-43b2-8a3e-6b766ae8b609)

## WSI级别分类(基于临床知识规则的启发式软投票机制)
![wsi-vote](https://github.com/user-attachments/assets/83751bc2-dd9a-4013-97be-797b4e34f439)


## 环境配置

    conda create -n CMSwinKAN python=3.8.20
    conda activate CMSwinKAN
    pip install -r requirements.txt

## 数据集组织
以下是默认的数据集组织方式（训练集 : 测试集 = 8 : 2）<br>

    dataset  
     ├── train
     │   ├── class1 
     │   ├── class2  
     │   └── ... 
     └── test
         ├── class1
         ├── class2
         └── ...
## 图像级别训练
 在`train_new.py`中修改`data-path`和 设置`num_classes`、`num_workers`(对于Windows系统, 建议设置为`0`)<br>

    python train_new.py

## 图像级别测试/评估

    python val.py

## WSI级别测试/评估(硬投票)

    python WSI_vote_hard.py

## WSI级别测试/评估(软投票)

    python WSI_vote.py

## 引用
### 部分代码引用自：

[Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

[efficient-kan](https://github.com/Blealtan/efficient-kan)

[ConDSeg](https://github.com/Mengqi-Lei/ConDSeg)

[SCKansformer](https://github.com/JustlfC03/SCKansformer)
