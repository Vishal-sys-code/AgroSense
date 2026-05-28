# AgroSense v2

**Multimodal Deep Learning for Precision Crop Recommendation**  
*Integrating Soil Vision, Nutrient Profiling, and Geospatial Intelligence*

---

## Overview

AgroSense v2 is a research-grade multimodal agricultural intelligence framework that unifies soil image classification, nutrient-aware crop recommendation, and geospatial representation learning into a single end-to-end pipeline.

The core architectural contribution is a **Cross-Modal Transformer (CMT)** fusion module that replaces naive feature concatenation with bidirectional cross-attention between visual soil embeddings and tabular nutrient representations — enabling each modality to inform the other during feature extraction rather than only at the decision boundary.

> **Reference paper:** [AgroSense: An Integrated Deep Learning System for Crop Recommendation via Soil Image Analysis and Nutrient Profiling](https://arxiv.org/abs/2509.01344)  
> **v2 Kaggle notebook:** [kaggle.com/code/rishabhhhme/agrosense-2-0](https://www.kaggle.com/code/rishabhhhme/agrosense-2-0)

---

## Research Scope

- Multimodal feature fusion for agricultural decision support  
- Soil image classification via convolutional and transformer architectures  
- Nutrient-aware crop recommendation with ensemble and neural models  
- Geospatial agricultural representation learning  
- Explainability through SHAP attribution and Grad-CAM saliency  

---

## Architecture

### Cross-Modal Transformer Fusion

```
Soil Image                         Nutrient Features (N, P, K, pH, T, H, R)
     │                                             │
EfficientNet-B0                          MLP Tabular Encoder
(patch-level features)                   (7 → 128 → 256)
     │                                             │
F_img ∈ R^(1×256)                      F_tab ∈ R^(1×256)
     │                                             │
     └──────────── Cross-Attention ───────────────┘
                  Tab → Img  |  Img → Tab
                         │
                   Fused Vector ∈ R^512
                    ┌──────────┐
                    │          │
              Crop Head    Soil Head
             (22 classes)  (7 classes)
```

### Soil Classification Module

| Architecture | Test Accuracy | F1-Score |
|---|---|---|
| Custom CNN | 88.9% | 87.5% |
| ResNet-18 | 89.8% | 88.9% |
| EfficientNet-B0 | 95.48% | 0.95 |
| Vision Transformer (ViT-Base) | 92.0% | 91.0% |

### Crop Recommendation Module

| Model | Test Accuracy | Modality |
|---|---|---|
| LightGBM | 98.64% | Tabular only |
| EfficientNet-B0 | 95.48% | Image only |
| **AgroSense v2 (CMT)** | **96.66%** | **Multimodal** |

> The multimodal model achieves superior calibration and out-of-distribution robustness compared to unimodal baselines, as validated by ablation studies with modality dropout.

---

## Repository Structure

```
AgroSense/
├── Datasets/
│   ├── normalized_geospatial_patches.npy
│   ├── Soil Image Dataset/
│   └── Soil types/
│
├── Saved Models/
│   ├── best_agrosense2.pth           # CMT fusion checkpoint
│   ├── best_soil_model.pth           # EfficientNet soil classifier
│   ├── crop_recommendation_lgb_model.pkl
│   ├── crop_recommendation_xgb_model.pkl
│   ├── crop_recommendation_mlp_model.pth
│   ├── crop_recommendation_tabnet_model.pkl
│   ├── efficientnet_b0_model.pth
│   ├── resnet18_model.pth
│   └── vit_model.pth
│
├── agrosense_v2.ipynb
├── app.py
├── path.py
├── requirements.txt
└── README.md
```

---

## Data Sources

| Dataset | Description | Source |
|---|---|---|
| SoilGrids v2 | 250m global soil property rasters (N, pH, SOC, clay, sand, silt, bulk density) | ISRIC |
| WoSIS 2023 | World Soil Information Service snapshot | ISRIC |
| OpenLandMap | Geospatial soil property samples | Zenodo |
| FAO GAEZ v4 | Agro-climatic and yield data | FAO |
| Kaggle Soil Images | 7-class soil image dataset with CycleGAN augmentation | Kaggle |
| Crop Recommendation CSV | 2,200 samples across 22 crops with NPK + climate features | Kaggle |

---

## Installation

```bash
git clone https://github.com/rishabhh-me/AgroSense.git
cd AgroSense
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Streamlit Application

```bash
streamlit run app.py
```

Supports soil image upload, soil type classification, and nutrient-aware crop recommendation via the full CMT inference pipeline.

### Research Notebook

```bash
jupyter notebook agrosense_v2.ipynb
```

Contains end-to-end pipelines for data preprocessing, model training, ablation studies, SHAP attribution analysis, and Grad-CAM saliency generation.

---

## Explainability

**SHAP Attribution (LightGBM)**  
Global feature importance ranked by mean absolute SHAP value across 220 test samples:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Rainfall | 0.353 |
| 2 | Humidity | 0.353 |
| 3 | Phosphorus (P) | 0.276 |
| 4 | Nitrogen (N) | 0.202 |
| 5 | Potassium (K) | 0.187 |
| 6 | Temperature | 0.060 |
| 7 | pH | 0.037 |

**Grad-CAM (EfficientNet-B0)**  
Class-discriminative saliency maps are generated for each of the 7 soil classes, highlighting texture regions — granular patterns in laterite, crack structures in yellow soil, uniform dark saturation in black soil — that drive classification decisions.

---

## Soil Classes

Alluvial · Arid · Black · Laterite · Mountain · Red · Yellow

## Crop Classes

Apple · Banana · Blackgram · Chickpea · Coconut · Coffee · Cotton · Grapes · Jute · Kidneybeans · Lentil · Maize · Mango · Mothbeans · Mungbean · Muskmelon · Orange · Papaya · Pigeonpeas · Pomegranate · Rice · Watermelon

---

## Technology Stack

**Frameworks:** Python · PyTorch · Streamlit  
**ML/DL:** EfficientNet · ResNet · Vision Transformer · LightGBM · XGBoost · TabNet  
**Geospatial:** GDAL · rasterio · Google Earth Engine  
**Explainability:** SHAP · Grad-CAM  
**Infrastructure:** Kaggle (GPU T4 x2) · Git LFS · Jupyter  

---

## Future Directions

- Satellite imagery integration via Sentinel-2 spectral bands  
- Temporal soil property modelling for seasonal drift  
- Real-time IoT sensor fusion for field deployment  
- Transformer-based agricultural foundation model pretraining  
- Knowledge distillation for mobile edge inference  

---

## Authors

**Original AgroSense**  
Vishal Pandey — pandeyvishal.mlprof@gmail.com

**AgroSense v2**  
Rishav Tewari — AI/ML Research Engineer  
[github.com/rishabhh-me](https://github.com/rishabhh-me) · rishavtewari.research@gmail.com

---

## License

This project follows the license of the original AgroSense repository.
