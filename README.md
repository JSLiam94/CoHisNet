CoHisNet
=========
**``Read this in other languages: ``English|[中文](README_zh.md).**
--------
## Paper
##### [Towards Accurate and Interpretable Neuroblastoma Diagnosis via Contrastive Multi-scale Pathological Image Analysis](https://arxiv.org/abs/2504.13754)
## The origin of the name
### Contrast Multi-Scale Adaptive Swin KANsformer for pathological images classification
### Contrastive Histopathology Network

## Patch-level classification
![CoHisNet](https://github.com/user-attachments/assets/24950147-6ba7-42ff-89e0-6ed6fc3d0bba)

### WSI-level classification(a heuristic Soft Voting classification process based on clinical knowledge rules)
![wsi-vote](https://github.com/user-attachments/assets/c8223b14-4243-427d-92e3-1638f218110c)

## Private Dataset · Collection and Preprocessing
![data](https://github.com/user-attachments/assets/efadaa36-172e-4d36-85fc-6adbe35390c6)

## Environment

    conda create -n CoHisNet python=3.8.20
    conda activate CoHisNet
    pip install -r requirements.txt

## Dataset
The default organization method of the dataset(train : test = 8 : 2)<br>

    dataset  
     ├── train
     │   ├── class1 
     │   ├── class2  
     │   └── ... 
     └── test
         ├── class1
         ├── class2
         └── ...
## Patch-level(or image) training
Modify the `data-path` and set `num_classes`、`num_workers` (for Windows system, it is recommended to set it to `0`) in train_new.py<br>

    python train_new.py

## Patch-level evaluation

    python val.py

## WSI-level evaluation(hard vote)

    python WSI_vote_hard.py

## WSI-level evaluation(soft vote)

    python WSI_vote.py
  
## References
### Some of the codes are borrowed from:

[Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

[efficient-kan](https://github.com/Blealtan/efficient-kan)

[ConDSeg](https://github.com/Mengqi-Lei/ConDSeg)

[SCKansformer](https://github.com/JustlfC03/SCKansformer)
