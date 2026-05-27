# 🌱 AgroSense v2: Multimodal Agricultural Intelligence System

<p align="center">
  <b>AI-Powered Precision Agriculture using Deep Learning, Geospatial Intelligence, and Multimodal Data Fusion</b>
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

AgroSense v2 is a next-generation multimodal agricultural intelligence framework designed for precision farming, soil understanding, and intelligent crop recommendation.

The system integrates:

- 🌍 Geospatial agricultural data
- 🧠 Deep learning-based soil intelligence
- 🌱 Crop recommendation systems
- 🛰️ Remote sensing representations
- 📊 Multimodal feature fusion pipelines

to provide an end-to-end AI-driven decision support ecosystem for smart agriculture.

Unlike traditional crop recommendation systems that rely purely on tabular nutrient inputs, AgroSense v2 combines visual, numerical, and geospatial modalities into a unified inference pipeline.

---

# 🚀 What's New in AgroSense v2

AgroSense v2 significantly extends the original framework with:

## ✅ Geospatial Intelligence Integration
- Added normalized geospatial agricultural patch datasets
- Support for spatial feature experimentation
- Remote sensing-oriented preprocessing workflows

## ✅ Enhanced Research Notebook
- Added advanced experimentation notebook:
  
```text
agrosense_v2.ipynb
```

- Modular experimentation workflows
- End-to-end research pipelines
- Rapid prototyping environment for agricultural AI

## ✅ New Deep Learning Models
Additional pretrained checkpoints added for:
- soil intelligence
- multimodal experimentation
- agricultural prediction tasks

## ✅ Improved Research Reproducibility
- Better repository organization
- Git LFS integration for large-scale datasets
- Structured experimentation workflows

---

# 🧠 System Architecture

AgroSense v2 follows a multimodal architecture composed of three primary subsystems:

## 1️⃣ Soil Intelligence Module

This module performs automated soil understanding using deep learning.

### Supported Architectures
- Custom CNN
- ResNet18
- EfficientNet-B0
- Vision Transformer (ViT)

### Pipeline
```text
Soil Image → Feature Extraction → Soil Classification → Encoded Soil Representation
```

---

## 2️⃣ Crop Recommendation Module

This subsystem predicts optimal crops using:
- soil composition
- environmental parameters
- multimodal encoded features

### Features Used
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- pH
- Temperature
- Humidity
- Rainfall
- Encoded soil type

### Models
- Multi-Layer Perceptron (MLP)
- XGBoost
- LightGBM
- TabNet

---

## 3️⃣ Geospatial Intelligence Module (v2)

AgroSense v2 introduces geospatial agricultural representations for:
- spatial analysis
- remote sensing experimentation
- land intelligence workflows
- geospatial feature learning

### Added Dataset
```text
Datasets/normalized_geospatial_patches.npy
```

Potential applications include:
- precision agriculture
- crop health monitoring
- land-use intelligence
- agricultural segmentation
- satellite-assisted prediction systems

---

# 📓 Research Notebook

## AgroSense v2 Notebook

```text
agrosense_v2.ipynb
```

This notebook contains:
- deep learning experimentation
- geospatial workflows
- multimodal training pipelines
- agricultural AI research experiments
- visualization and evaluation workflows

### Kaggle Notebook
https://www.kaggle.com/code/rishabhhhme/agrosense-2-0

---

# 🧬 Added Pretrained Models

## 1️⃣ best_agrosense2.pth

Enhanced multimodal AgroSense checkpoint for:
- agricultural intelligence tasks
- multimodal inference
- crop recommendation workflows
- integrated experimentation

---

## 2️⃣ best_soil_model.pth

Dedicated soil classification model for:
- soil type prediction
- soil image analysis
- agricultural soil intelligence

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

## Launch Streamlit Web Application

```bash
streamlit run app.py
```

The application enables:
- soil image upload
- soil classification
- nutrient profiling
- crop recommendation
- multimodal agricultural inference

---

# 📊 Running Research Experiments

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
agrosense_v2.ipynb
```

---

# 🔬 Experimental Focus Areas

AgroSense v2 explores several advanced agricultural AI domains:

- Soil Image Classification
- Multimodal Learning
- Precision Agriculture
- Geospatial Intelligence
- Remote Sensing Integration
- Agricultural Decision Support Systems
- Deep Learning for Smart Farming
- Spatial Feature Learning

---

# 📈 Research Contributions

AgroSense v2 contributes toward next-generation agricultural AI through:

## ✅ Multimodal Agricultural Intelligence
Combines visual, numerical, and geospatial information into a unified prediction framework.

## ✅ Geospatial AI Integration
Introduces spatial agricultural representations into crop intelligence workflows.

## ✅ Real-Time Agricultural Decision Support
Deployable Streamlit-based intelligent farming assistant.

## ✅ Scalable AI Research Framework
Provides a modular environment for future agricultural AI experimentation.

---

# 🛠️ Tech Stack

## Core Frameworks
- Python
- PyTorch
- Streamlit
- NumPy
- Pandas
- OpenCV
- Scikit-learn

## Deep Learning
- CNNs
- EfficientNet
- ResNet
- Vision Transformers

## Research Tools
- Jupyter Notebook
- Kaggle
- Git LFS

---

# 🌍 Future Directions

Planned future enhancements include:

- Satellite imagery integration
- Temporal agricultural forecasting
- Transformer-based agricultural foundation models
- Explainable AI for crop recommendation
- Distributed training pipelines
- Cloud deployment infrastructure
- Agricultural LLM integration
- Real-time IoT sensor fusion

---

# 📜 Research References

## Original Research Paper
https://arxiv.org/abs/2509.01344

## Original Notebook
https://www.kaggle.com/code/tmleyncodes/research-work-agrosense-agricultural-ai

## AgroSense v2 Notebook
https://www.kaggle.com/code/rishabhhhme/agrosense-2-0

---

# 🤝 Contributions

Contributions, research collaborations, and feature suggestions are welcome.

If you'd like to contribute:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

# 📄 License

This project follows the original repository license.

---

# 👨‍💻 Author

### Rishabh Mishra
AI/ML Research Engineer • Agricultural AI • Geospatial Intelligence • Deep Learning

GitHub:
https://github.com/rishabhh-me

---

# ⭐ Acknowledgements

Special thanks to:
- the original AgroSense authors,
- open-source AI communities,
- agricultural AI researchers,
- and the geospatial deep learning ecosystem.

---

<p align="center">
  <b>AgroSense v2 — Advancing AI-Driven Precision Agriculture Through Multimodal Intelligence</b>
</p>
