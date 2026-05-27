# 🌱 AgroSense v2: Multimodal Agricultural Intelligence Framework for Precision Agriculture

<p align="center">
  <b>Deep Learning • Geospatial Intelligence • Multimodal AI • Precision Farming</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg"/>
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red.svg"/>
  <img src="https://img.shields.io/badge/Streamlit-WebApp-orange.svg"/>
  <img src="https://img.shields.io/badge/Research-AgriculturalAI-green.svg"/>
  <img src="https://img.shields.io/badge/Status-Active-success.svg"/>
</p>

---

# 📌 Overview

AgroSense v2 is a multimodal agricultural intelligence system designed to advance precision agriculture through the integration of deep learning, geospatial representations, and environmental feature fusion.

The framework combines:

- soil image intelligence,
- nutrient-aware crop recommendation,
- geospatial agricultural representations,
- and multimodal learning pipelines

to deliver an end-to-end AI-driven agricultural decision support ecosystem.

Unlike conventional crop recommendation systems that rely solely on tabular nutrient data, AgroSense v2 incorporates visual and spatial representations to improve contextual agricultural understanding and extensibility toward next-generation remote sensing workflows.

---

# 🚀 Key Features

## 🧠 Soil Intelligence Module
Advanced deep learning models for automated soil classification using soil imagery.

### Supported Architectures
- Custom CNN
- ResNet18
- EfficientNet-B0
- Vision Transformer (ViT)

### Capabilities
- Soil type prediction
- Visual soil understanding
- Feature extraction for multimodal fusion
- Agricultural image intelligence

---

## 🌱 Crop Recommendation Engine

A multimodal crop recommendation subsystem integrating:

- soil classification outputs,
- environmental parameters,
- and nutrient profiling.

### Features Used
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- pH
- Temperature
- Humidity
- Rainfall
- Encoded soil representations

### Implemented Models
- Multi-Layer Perceptron (MLP)
- XGBoost
- LightGBM
- TabNet

---

## 🛰️ Geospatial Intelligence Integration (v2)

AgroSense v2 introduces geospatial agricultural representations for spatial experimentation and remote sensing-oriented agricultural AI research.

### Added Dataset
```text
Datasets/normalized_geospatial_patches.npy
```

### Applications
- Precision agriculture
- Spatial crop intelligence
- Remote sensing experimentation
- Agricultural land analysis
- Geospatial feature learning

---

# 📓 AgroSense v2 Research Notebook

```text
agrosense_v2.ipynb
```

The v2 notebook provides:
- multimodal experimentation pipelines,
- geospatial preprocessing workflows,
- model experimentation,
- visualization utilities,
- and reproducible research infrastructure.

### Kaggle Notebook
https://www.kaggle.com/code/rishabhhhme/agrosense-2-0

---

# 🧬 Added Pretrained Models

## `best_agrosense2.pth`

Enhanced multimodal AgroSense checkpoint supporting:
- integrated agricultural inference,
- multimodal experimentation,
- crop recommendation workflows,
- and agricultural prediction research.

---

## `best_soil_model.pth`

Dedicated deep learning checkpoint for:
- soil image classification,
- soil intelligence workflows,
- and agricultural image analysis.

---

# 🏗️ System Architecture

AgroSense v2 follows a modular multimodal architecture composed of three primary subsystems.

---

## 1️⃣ Soil Intelligence Pipeline

```text
Soil Image
    ↓
Image Preprocessing
    ↓
Deep Feature Extraction
    ↓
Soil Classification
    ↓
Encoded Soil Representation
```

### Preprocessing
- Image resizing
- Normalization
- Feature standardization
- Data augmentation

---

## 2️⃣ Crop Recommendation Pipeline

```text
Environmental Features
        +
Encoded Soil Features
        ↓
Multimodal Feature Fusion
        ↓
ML/DL Recommendation Models
        ↓
Crop Recommendation
```

---

## 3️⃣ Geospatial Intelligence Pipeline

```text
Geospatial Agricultural Patches
            ↓
Normalization & Processing
            ↓
Spatial Representation Learning
            ↓
Agricultural Intelligence Workflows
```

---

# 📂 Repository Structure

```text
AgroSense/
│
├── Datasets/
│   ├── normalized_geospatial_patches.npy
│   ├── Soil Image Dataset/
│   ├── Soil types/
│   └── ...
│
├── Saved Models/
│   ├── best_agrosense2.pth
│   ├── best_soil_model.pth
│   ├── crop_recommendation_mlp_model.pth
│   ├── crop_recommendation_xgb_model.pkl
│   ├── crop_recommendation_lgb_model.pkl
│   ├── crop_recommendation_tabnet_model.pkl
│   ├── efficientnet_b0_model.pth
│   ├── resnet18_model.pth
│   ├── vit_model.pth
│   └── ...
│
├── agrosense_v2.ipynb
├── app.py
├── path.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/rishabhh-me/AgroSense.git
cd AgroSense
```

---

## Create Virtual Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Launch Streamlit Application

```bash
streamlit run app.py
```

The web application supports:
- soil image upload,
- soil classification,
- nutrient-aware crop recommendation,
- and multimodal agricultural inference.

---

# 🔬 Running Research Experiments

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
agrosense_v2.ipynb
```

---

# 📊 Research Domains

AgroSense v2 explores multiple advanced agricultural AI domains:

- Precision Agriculture
- Multimodal Learning
- Soil Intelligence
- Geospatial AI
- Remote Sensing
- Agricultural Decision Systems
- Deep Learning for Smart Farming
- Spatial Representation Learning

---

# 🧪 Experimental Focus

Current experimentation includes:
- soil image classification,
- multimodal feature fusion,
- agricultural deep learning,
- geospatial representation learning,
- crop recommendation systems,
- and intelligent agricultural inference pipelines.

---

# 📈 Research Contributions

## ✅ Multimodal Agricultural Intelligence
Integrates visual, numerical, and geospatial representations into a unified agricultural AI framework.

## ✅ Geospatial AI Integration
Introduces spatial agricultural representations into crop intelligence workflows.

## ✅ End-to-End Agricultural Inference
Provides a deployable Streamlit-based intelligent farming assistant.

## ✅ Scalable Research Infrastructure
Supports future experimentation in:
- satellite intelligence,
- agricultural transformers,
- explainable AI,
- and spatial deep learning.

---

# 🛠️ Technology Stack

## Core Frameworks
- Python
- PyTorch
- Streamlit
- NumPy
- Pandas
- OpenCV
- Scikit-learn

## Deep Learning Architectures
- CNNs
- EfficientNet
- ResNet
- Vision Transformers

## Research Tooling
- Jupyter Notebook
- Kaggle
- Git LFS

---

# 🌍 Future Directions

Planned future extensions include:

- satellite imagery integration,
- transformer-based agricultural foundation models,
- explainable crop recommendation systems,
- temporal agricultural forecasting,
- distributed training infrastructure,
- real-time IoT sensor fusion,
- and cloud-native deployment pipelines.

---

# 📚 Research References

## Original Research Paper
https://arxiv.org/abs/2509.01344

## Original AgroSense Notebook
https://www.kaggle.com/code/tmleyncodes/research-work-agrosense-agricultural-ai

## AgroSense v2 Notebook
https://www.kaggle.com/code/rishabhhhme/agrosense-2-0

---

# 🤝 Contributions

Contributions, research collaborations, and feature proposals are welcome.

To contribute:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

# 📄 License

This project follows the original repository license.

---

# 👨‍💻 Authors

## Original AgroSense Author
### Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com

---

## AgroSense v2 Contributor & Maintainer
### Rishav Tewari

AI/ML Research Engineer • Agricultural AI • Geospatial Intelligence • Deep Learning

GitHub:
https://github.com/rishabhh-me

Email:
rishavtewari.research@gmail.com

---

# 📬 Contact

For research discussions, collaborations, or project inquiries:

### Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com

### Rishav Tewari
Email: rishavtewari.research@gmail.com

---

# ⭐ Acknowledgements

Special thanks to:
- the original AgroSense research contributors,
- open-source AI communities,
- geospatial deep learning researchers,
- and the precision agriculture ecosystem.

---

<p align="center">
  <b>AgroSense v2 — Advancing Precision Agriculture Through Multimodal Artificial Intelligence</b>
</p>
