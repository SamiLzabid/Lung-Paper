# FeLASwin-Net: Dual Modality Lung Disease Classification

Source code and implementation details for our journal paper on lung disease classification from **CT scans** and **chest X-ray** images.

**FeLASwin-Net** is a lightweight hybrid architecture that combines a convolutional front-end (residual blocks with a **FETLE** module and **AMSFA** downsampling) with shifted-window Swin Transformer blocks. The same pipeline is applied to two imaging modalities (CT and chest X-ray) by changing only the dataset.

---

## Table of Contents

- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Dataset Information](#dataset-information)
- [Pipeline Overview](#pipeline-overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Web App](#web-app)
- [Citation](#citation)
- [License](#license)

---

## Highlights

- Lightweight model: about **2.81 M parameters**, **0.634 GFLOPs**, **10.93 MB** (profiled at 224×224 input).
- Two novel modules: **FETLE** (Feature Enhancement via Texture and Lesion-Edge extraction) and **AMSFA** (Asymmetric Multi-Scale Feature Aggregation).
- Full evaluation protocol: classification metrics, corruption robustness, uncertainty and calibration, 5-fold cross-validation, ablation, statistical testing, and explainable AI.
- Same code evaluated on both **lung CT** and **chest X-ray** data.
- Interactive web app for inference with explanations.

> Please verify the expansions of the FETLE and AMSFA acronyms against the paper and edit them if needed.

---

## Repository Structure

```
.
├── codes/
│   ├── felaswin_net.py                  # FeLASwin-Net model (FETLE, AMSFA, Swin blocks)
│   ├── Training_Evaluation_Fuctions.py  # Training loop, metrics, robustness, MC Dropout/TTA, calibration
│   ├── Lung_CT.ipynb                    # End-to-end pipeline on the lung CT dataset
│   ├── Chest_Xray.ipynb                 # Same pipeline on the chest X-ray dataset
│   ├── my_dataset.py                    # Custom PyTorch Dataset (cv2 loader)
├── web_app/                             # Web application for inference (see Web App section)
└── README.md
```

`Lung_CT.ipynb` and `Chest_Xray.ipynb` contain the same implementation, applied to different datasets. Only the dataset path, preprocessing output folder, and class names differ.

---

## Dataset Information

The datasets used in this study are publicly available through Kaggle and Mendeley Data.

**Computed Tomography (CT)**

| Dataset | Link |
|---|---|
| IQ-OTH/NCCD – Lung Cancer Dataset | https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset |
| An Image Dataset of Advanced Imaging Techniques for Lung Cancer Diagnosis | https://data.mendeley.com/datasets/zr4fddh833/1 |

**Chest Radiography (X-ray)**

| Dataset | Link |
|---|---|
| Chest X-Ray (Covid-19 & Pneumonia) | https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia |
| COVID-19 Radiography Database | https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data |


Update `data_dir` (raw images), `output_dir` (preprocessed images), and `expanded_train_dir` (augmented training set) at the top of the notebooks to match your local paths.

---

## Pipeline Overview

### 1. Data Loading
Images are read from class sub-folders into a DataFrame of file paths and labels. A custom PyTorch `Dataset` (`CustomImageDataset`) loads each image with OpenCV (BGR→RGB). Class distributions and sample images are visualized per class.

### 2. Preprocessing
Contrast enhancement is applied once, offline, to every image:
- Percentile-based contrast stretching computed on non-background pixels only (20th–98th percentile), so black padding does not skew the range.
- Gamma correction (γ = 1.4) to suppress mid-gray haze.
- Unsharp-style sharpening to emphasize vessel and edge structures.
- Conversion to 3-channel grayscale for the network.

Images are resized to **224×224** and normalized with ImageNet mean and std.

### 3. Data Splitting
Stratified split with `random_state=42`: **80% train / 10% validation / 10% test**.

### 4. Augmentation
Offline augmentation is applied to the **training set only**, bringing every class up to a fixed size (2,000 images per class) to address class imbalance. Validation and test sets are never augmented. The augmentations (Albumentations) are:
- `ShiftScaleRotate` (shift 0.1, scale 0.1, rotation ≤ 15°)
- `ElasticTransform`
- `RandomBrightnessContrast`
- `GaussNoise`

A with/without augmentation comparison is included in the notebooks.

### 5. Model Training
- Trained **from scratch** (no pretrained weights), batch size 32, mixed precision (AMP).
- Loss: cross-entropy with label smoothing 0.1.
- Optimizer: AdamW (lr 1e-4, weight decay 0.04, with no decay on biases, norms, and LayerScale parameters).
- Scheduler: 5-epoch linear warmup followed by cosine annealing (to 1e-6), up to 30 epochs.
- Early stopping on validation loss (patience 5, min delta 0.001).
- Training and validation curves plus training and inference time are reported.

### 6. Evaluation
Reported on the held-out test set:
- Accuracy, macro Precision, Recall, F1, Specificity, MCC, Cohen's Kappa, weighted PR-AUC
- Confusion matrix and per-class ROC and Precision–Recall curves (600 dpi PNG, with PDF/SVG copies)
- Error analysis: misclassified samples, confidence box plot (correct vs. misclassified), and t-SNE feature-space plot

### 7. Robustness and Reliability Testing
**Corruption robustness:** four clinically realistic corruptions at four severity levels each:
- Gaussian noise
- Gaussian blur
- Brightness shift
- JPEG compression

Accuracy, macro-F1, and MCC are reported per severity, as the mean across severities, and as the drop from the clean baseline.

**Uncertainty and calibration:**
- **MC Dropout** (30 stochastic passes) and **Test-Time Augmentation** (5 views).
- Predictive entropy decomposed into aleatoric and epistemic components.
- Calibration metrics: **ECE, MCE, Brier score, NLL**, plus reliability diagrams.

### 8. Cross-Validation
**5-fold stratified cross-validation** over the full dataset, with augmentation re-run inside each fold using only that fold's training data to avoid leakage. Results are reported as per-fold values and mean ± std. The same protocol is run for baseline models, and **McNemar's test** is applied to pooled out-of-fold predictions to test whether the difference between FeLASwin-Net and DenseNet169 is statistically significant.

### 9. Ablation Study
- **Module ablation:** full model vs. w/o FETLE, w/o AMSFA, and baseline (without both). AMSFA is replaced by a standard strided convolution.
- **Augmentation ablation:** performance with and without offline augmentation.
- **Baseline comparison:** DenseNet169, ResNeXt101, EfficientNet-B4, ConvNeXtV2-tiny, LeViT-192, SwinV2-tiny, MaxViT-tiny.
- **Efficiency comparison:** parameters, GFLOPs (via `fvcore`), model size, GPU memory, and inference time.

### 10. Explainable AI (XAI)
- **Methods:** Grad-CAM++, Score-CAM, and Transition Matrix Heatmap, applied to the last Swin block (token sequences are reshaped to a spatial map).
- **Qualitative:** side-by-side overlay grid per class.
- **Quantitative faithfulness:** Deletion and Insertion AUC per class and overall, exported to CSV.

---

## Model Architecture

```
Input 224×224
 └─ Stem (7×7 conv, BN, GELU, max-pool)                      → 56×56
     ├─ Stage 1: Residual-FETLE block → AMSFA (48→96)        → 28×28
     ├─ Stage 2: Residual-FETLE block → AMSFA (96→128)       → 14×14
     ├─ Stage 3: 4× Swin blocks (window 7, shifted)          → 14×14
     ├─ Stage 4: AMSFA (128→256) → 2× Swin blocks            → 7×7
     └─ LayerNorm → global average pooling → Dropout → Linear classifier
```

- **FETLE:** parallel depthwise texture, dilated edge, and high-frequency branches with learnable softmax-weighted fusion.
- **AMSFA:** four-path downsampling (strided 1×1, depthwise-separable, max-pool, and asymmetric 1×k / k×1) with learnable path weights.
- **Swin blocks:** native window attention with relative position bias, shifted windows, LayerScale, DropPath, and attention dropout.

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch torchvision timm albumentations opencv-python pillow \
            numpy pandas scipy scikit-learn statsmodels matplotlib seaborn \
            tqdm fvcore jupyter
```

A CUDA-capable GPU is recommended (the notebooks assume one for training).

---

## Usage

1. Download a dataset (see [Dataset Information](#dataset-information)) and arrange it in class sub-folders.
2. Open `codes/Lung_CT.ipynb` (or `codes/Chest_Xray.ipynb`) and set the dataset paths.
3. Run the notebook top to bottom: preprocessing → splitting → augmentation → training → evaluation → robustness and uncertainty → ablation → cross-validation → XAI.

Quick check of the model:

```python
from felaswin_net import FeLASwinNet

model = FeLASwinNet(num_classes=3, drop_path_rate=0.1, attn_drop_rate=0.1)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
```

---

## Web App
 
The `web_app/` folder contains a [Streamlit](https://streamlit.io/) application that runs the trained FeLASwin-Net models on uploaded images and explains each prediction.
 
### Interface
 
| Lung CT Scan | Chest X-Ray with Grad-CAM |
|:---:|:---:|
| ![Lung CT interface](<img width="931" height="878" alt="WebAppSS" src="https://github.com/user-attachments/assets/999421de-1a4a-4d08-88e1-fa27d996483e" />) | ![Chest X-ray interface with Grad-CAM](<img width="927" height="909" alt="WebAppSS1" src="https://github.com/user-attachments/assets/872c7e74-3bb9-46e7-b89a-a0334fd0ff11" />) |
 
### Features
 
- **Modality selection:** choose *Lung CT Scan* (Benign, Malignant, Normal) or *Chest X-Ray* (COVID-19, Pneumonia, Normal). The matching model is loaded automatically and cached.
- **Image upload:** PNG, JPG and JPEG, with an input preview and a one-click re-upload.
- **Diagnosis report:** predicted class, softmax probability for every class, a color-coded status bar (green, amber, red), and a short patient-education overview of the predicted condition.
- **Grad-CAM explanation:** toggle button that overlays a heat map of the regions that contributed most to the prediction, with the target class and layer name shown. By default the last spatial feature layer with a usable gradient is selected automatically. To pin a layer, set `GRADCAM_TARGET_LAYER` in `app.py`.
- **Modality-specific preprocessing:** the CT contrast enhancement from the training pipeline is applied to CT uploads only. X-ray images are used as uploaded. Both are resized to 224x224 and normalized with ImageNet statistics.
### Run locally
 
```bash
cd web_app
pip install streamlit torch torchvision opencv-python pillow numpy
streamlit run app.py
```
 
A recent Streamlit release is required (the app uses keyed containers and Material icons on buttons). A GPU is optional, since the app falls back to CPU automatically.
 
Place the trained weights in `web_app/models/` using these exact file names:
 
```
web_app/models/felaswin_ct.pth
web_app/models/felaswin_xray.pth
```
 
If a weight file is missing, the app shows a warning and runs with untrained weights (UI demonstration only, so predictions are meaningless).

> **Medical disclaimer:** the app is a research demo, not a medical device. Its output must be confirmed by a licensed radiologist.
