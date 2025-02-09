# Brain-Tumor-Classification-using-Deep-Learning

## Description:

Brain Tumor Classification from Mutlisequence MRI (T1, T1C and T2) and Mutlimodal CT and MRI using EfficientNetV2B0 with Mutliheaded Self Attention and Hyperparameter Fine-Tuning

> Methodology

- Implementation of a novel framework for classifying various kinds of brain tumors and healthy patients from structural MRI scans of T1, T1C and T2 sequences as well CT scans.
- In the first stage, a pre-trained EfficientNetV2 architecture has been used followed by Mutli-Head Self Attention Mechanism on the extracted, high-dimensional sequential feature maps.
- Global Average Pooling, Batch Normalization, L1, L2 Regularization and Dropout along with fine-tuned hyperparameters have been applied before mutli-class classification through softmax activation function.

> Datasets used:
- **[Brain Tumor MRI Images 44 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)**
- **[Brain Tumor MRI Images 17 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes)**
- **[Brain tumor multimodal image (CT & MRI)](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri)**

> Workflow Used:

![Workflow of the proposed framework for brain tumor classification](https://github.com/user-attachments/assets/d889103d-72c8-43f6-ad19-196e4015311a)
![Classification Task](https://github.com/user-attachments/assets/b15931ac-f4c8-4ce4-b1ab-3a64d3134719)

> Program Files:
- [Data Preprocessing](https://github.com/ksatrajit0/Brain-Tumor-Classification-using-Deep-Learning/blob/main/data_preprocessing.py)
- [Model Architecture](https://github.com/ksatrajit0/Brain-Tumor-Classification-using-Deep-Learning/blob/main/model.py)
- [Training](https://github.com/ksatrajit0/Brain-Tumor-Classification-using-Deep-Learning/blob/main/training.py)
- [Evaluation](https://github.com/ksatrajit0/Brain-Tumor-Classification-using-Deep-Learning/blob/main/evaluation.py)
- [Grad-CAM Analysis](https://github.com/ksatrajit0/Brain-Tumor-Classification-using-Deep-Learning/blob/main/grad_cam_analysis.py)
- [Final Jupyter Notebook Version](https://github.com/ksatrajit0/Brain-Tumor-Classification-using-Deep-Learning/blob/main/brat-classfcn-mri-mutli-seqfusn-attention.ipynb)

### Importance of Project:
- The promising results achieved underscore the potential of our framework’s robust nature and generalization capabilities across various modalities.
- Assist medical professionals in making precise diagnoses and, ultimately enhance patient outcomes.

## Credit(s) and Acknowledgement:

Supervisor: **[Dr. Pawan Kumar Singh](https://scholar.google.com/citations?user=LctgJHoAAAAJ&hl=en&oi=ao)**

### Paper:
> It'd be great if you could cite our paper (under review) if this code has been helpful to you.
 
> Thank you very much!

