# CMSwinKAN
## Contrast Multi-Scale Adaptive Swin KANsformer
![image](https://github.com/user-attachments/assets/e374c837-91f7-4d56-9799-2116da9523e7)

# Environment
  python > 3.8
  torch > 2.0
  

# Dataset
The default organization method of the dataset<br>

    dataset  
     ├── train
     │   ├── class1 
     │   ├── class2  
     │   └── ... 
     └── test
         ├── class1
         ├── class2
         └── ...
# train
Modify the data-path and set num_classes、num_workers (for Windows system, it is recommended to set it to 0) in train_new.py<br>

    python train_new.py
# Evaluation
  python val.py
# References
## Some of the codes are borrowed from:

[Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

[efficient-kan](https://github.com/Blealtan/efficient-kan)

[ConDSeg](https://github.com/Mengqi-Lei/ConDSeg)

[SCKansformer](https://github.com/JustlfC03/SCKansformer)
