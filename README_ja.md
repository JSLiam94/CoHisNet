CoHisNet
=========
**``Read this in other languages: ``[English](README.md)|[中文](README_zh.md)|日本語.**
--------
## 論文
#### [対比駆動の多尺度病理画像解析による正確で解釈可能な神経芽細胞腫診断](https://arxiv.org/abs/2504.13754)

## パッチレベル分類
![CoHisNet](https://github.com/user-attachments/assets/24950147-6ba7-42ff-89e0-6ed6fc3d0bba)

### WSIレベル分類（臨床知識ルールに基づくヒューリスティックなソフト投票分類プロセス）
![wsi-vote](https://github.com/user-attachments/assets/c8223b14-4243-427d-92e3-1638f218110c)

## 環境設定

    conda create -n CoHisNet python=3.8.20
    conda activate CoHisNet
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
