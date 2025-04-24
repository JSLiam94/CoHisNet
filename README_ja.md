CMSwinKAN
=========
**``Read this in other languages: ``[English](README.md)|[中文](README_zh.md)|日本語.**
--------
## 論文
#### [対比駆動の多尺度病理画像解析による正確で解釈可能な神経芽細胞腫診断](https://arxiv.org/abs/2504.13754)
## 名前の由来
### 病理画像分類のための対比多尺度適応型Swin KANsformer

## パッチレベル分類
![CMSwinKAN](https://github.com/user-attachments/assets/531148e7-b1ce-4ee9-bf24-c13f0c6d70ac)

### WSIレベル分類（臨床知識ルールに基づくヒューリスティックなソフト投票分類プロセス）
![wsi-vote](https://github.com/user-attachments/assets/b9b13863-4054-41a8-b7a0-ab59e986f6ac)

## 環境設定

    conda create -n CMSwinKAN python=3.8.20
    conda activate CMSwinKAN
    pip install -r requirements.txt

## データセット
デフォルトのデータセットの組織方法（訓練：テスト = 8：2）<br>

    dataset  
     ├── train
     │   ├── class1 
     │   ├── class2  
     │   └── ... 
     └── test
         ├── class1
         ├── class2
         └── ...

## パッチレベル（または画像）トレーニング
train_new.pyで`data-path`を修正し、`num_classes`、`num_workers`を設定します（Windowsシステムの場合、`0`に設定することをお勧めします）<br>

    python train_new.py

## パッチレベル評価

    python val.py

## WSIレベル評価（ハード投票）

    python WSI_vote_hard.py

## WSIレベル評価（ソフト投票）

    python WSI_vote.py
  
## 参考文献
### 一部のコードは以下から引用しています：

[Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

[efficient-kan](https://github.com/Blealtan/efficient-kan)

[ConDSeg](https://github.com/Mengqi-Lei/ConDSeg)

[SCKansformer](https://github.com/JustlfC03/SCKansformer)
