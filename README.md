CMSwinKAN
=========
**``Read this in other languages: ``English|[中文](README_zh.md).**
--------
#### Paper
[Towards Accurate and Interpretable Neuroblastoma Diagnosis via Contrastive Multi-scale Pathological Image Analysis](https://arxiv.org/abs/2504.13754)
### The origin of the name
Contrast Multi-Scale Adaptive Swin KANsformer for pathological images classification

## Patch-level classification
![CMSwinKAN](https://github.com/user-attachments/assets/531148e7-b1ce-4ee9-bf24-c13f0c6d70ac)

### WSI-level classification(a heuristic Soft Voting classification process based on clinical knowledge rules)
![wsi-vote](https://github.com/user-attachments/assets/b9b13863-4054-41a8-b7a0-ab59e986f6ac)



## Environment

    conda create -n CMSwinKAN python=3.8.20
    conda activate CMSwinKAN
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
