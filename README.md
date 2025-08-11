# Brain Tumor Semantic Segmentation with U-Net (PyTorch)
Use of deep learning techniques for semantic segmentation of brain tumors in medical images using magnetic resonance imaging (MRI) data from a public dataset such as BRATS, available on platforms such as Kaggle.



## 📌 Goal

The aim of this project is to apply deep learning techniques for the **semantic segmentation of brain tumors** in medical imaging using **Magnetic Resonance Imaging (MRI)** data from the [BraTS 2020 dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data).  

The pipeline includes:
- Implementing and training a **Convolutional Neural Network** (preferably U-Net) for segmentation.
- Applying appropriate **pre- and post-processing**.
- Evaluating the model with **segmentation-specific metrics**.
- Using **PyTorch** for deep learning, **OpenCV** for image processing, and **Matplotlib** (or similar) for visualization.

---

## 📜 General Description

The complete workflow follows these stages:

1. **Load and preprocess** the MRI dataset from Kaggle.  
   - Split into training, validation, and test sets.  
   - Resize and normalize images.  
   - Prepare binary segmentation masks.

2. **Design and implement** a semantic segmentation network  
   - Preferably U-Net or similar encoder-decoder architecture.

3. **Train the model**  
   - Use segmentation loss functions (Dice Loss, Binary Cross-Entropy).  
   - Include early stopping or learning rate scheduling.

4. **Validate the model** on unseen data and visualize predictions.

5. **Evaluate performance**  
   - Metrics: Dice coefficient and Intersection over Union (IoU).

6. **Provide qualitative results**  
   - Overlay predicted masks on original images with proper color coding.

---

## 📊 Dataset

The [BraTS 2020 Training Data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) contains **pre-operative MRI scans** from multiple institutions with expert-annotated segmentation masks.  
Each case includes:
- **T1-weighted**
- **T1 with contrast enhancement (T1Gd)**
- **T2-weighted**
- **FLAIR**
- **Ground truth segmentation masks** for:
  - Enhancing tumor
  - Peritumoral edema
  - Necrotic core

For this project:
- Combine the three segmentation labels (edema = 2, enhancing tumor = 4, necrotic core = 1) into a **single binary mask**.

**Recommended split**:  
- 70% training  
- 15% validation  
- 15% testing  

| Brain Tumor Detection Image Examples |
|:-:|
| <img src="jupyter_imgs/brats.jpg" width="800px" height="300px"> |

---

## 🏗 Model Architecture

The model will be based on **U-Net**, a popular encoder-decoder architecture for biomedical image segmentation.  
Skip connections between encoder and decoder layers enable accurate localization of tumor regions.

| U-Net Schematic |
|:-:|
| <img src="jupyter_imgs/Unet.png" width="650px" height="360px"> |

---

## 📏 Evaluation Metrics

We will use **Dice Coefficient** and **IoU** to measure segmentation performance.

$$
\text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}
$$

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
$$

Where:
- **A** = predicted segmentation
- **B** = ground truth mask

---

## 📝 Assessment Criteria

### **Grade ≤ 6**
- Load & preprocess the FLAIR modality and binary masks.
- Implement a functional U-Net (or similar).
- Train the model & obtain predictions on validation/test set.

### **Grade ≤ 8** (plus previous)
- Generate qualitative results overlaying masks on original MRI slices.
- Apply **data augmentation** (random flips, rotations, elastic deformations).

### **Grade ≤ 10** (plus previous)
- Evaluate with Dice & IoU metrics.
- Plot Dice & IoU scores for test set.
- Analyze **3+ failure cases** and discuss error sources.

---

## ⚙️ Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- OpenCV
- Matplotlib
- NumPy
- Kaggle API (for dataset download)

Install dependencies:
```bash
pip install torch torchvision opencv-python matplotlib numpy kaggle

