# AgroSense v2  
### Multimodal Agricultural Intelligence for Precision Farming

AgroSense v2 is a research-oriented agricultural intelligence framework integrating deep learning, geospatial representations, and multimodal feature fusion for precision agriculture and intelligent crop recommendation.

The system combines:

- soil image understanding,
- nutrient-aware crop recommendation,
- geospatial agricultural representations,
- and multimodal learning pipelines

to create an extensible AI-driven decision support framework for smart farming applications.

Unlike conventional agricultural recommendation systems that rely solely on tabular nutrient data, AgroSense v2 incorporates visual and spatial representations for improved contextual understanding and scalability toward remote sensing and geospatial intelligence workflows.

---

## Research Scope

AgroSense v2 explores several interconnected domains within agricultural artificial intelligence:

- Soil Intelligence
- Precision Agriculture
- Geospatial AI
- Multimodal Learning
- Remote Sensing
- Agricultural Decision Systems
- Deep Learning for Smart Farming
- Spatial Representation Learning

---

## Core System Components

### Soil Intelligence Module

The soil intelligence subsystem performs automated soil understanding using deep convolutional and transformer-based architectures.

#### Implemented Architectures
- Custom CNN
- ResNet18
- EfficientNet-B0
- Vision Transformer (ViT)

#### Capabilities
- Soil type classification
- Visual feature extraction
- Agricultural image understanding
- Encoded soil representation generation

---

### Crop Recommendation Engine

The crop recommendation pipeline integrates environmental and learned visual representations to predict optimal crop selections.

#### Input Features
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- pH
- Temperature
- Humidity
- Rainfall
- Encoded soil representations

#### Implemented Models
- Multi-Layer Perceptron (MLP)
- XGBoost
- LightGBM
- TabNet

---

### Geospatial Intelligence Module

AgroSense v2 introduces geospatial agricultural representations for spatial experimentation and remote sensing-oriented workflows.

#### Added Dataset
```text
Datasets/normalized_geospatial_patches.npy
```

#### Research Applications
- Precision agriculture
- Spatial crop intelligence
- Agricultural land analysis
- Remote sensing experimentation
- Geospatial feature learning

---

# System Architecture

## Soil Intelligence Pipeline

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

---

## Crop Recommendation Pipeline

```text
Environmental Features
        +
Encoded Soil Features
        ↓
Multimodal Feature Fusion
        ↓
Recommendation Models
        ↓
Crop Recommendation
```

---

## Geospatial Intelligence Pipeline

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

# Research Notebook

```text
agrosense_v2.ipynb
```

The v2 notebook contains:
- multimodal experimentation workflows,
- geospatial preprocessing pipelines,
- model experimentation,
- evaluation workflows,
- and reproducible research utilities.

## Kaggle Notebook
https://www.kaggle.com/code/rishabhhhme/agrosense-2-0

---

# Added Pretrained Models

## `best_agrosense2.pth`

Enhanced multimodal AgroSense checkpoint for:
- agricultural intelligence workflows,
- multimodal inference,
- integrated experimentation,
- and crop recommendation research.

---

## `best_soil_model.pth`

Dedicated soil intelligence checkpoint for:
- soil classification,
- agricultural image analysis,
- and visual soil representation learning.

---

# Repository Structure

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

# Installation

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

# Running the Project

## Launch Streamlit Application

```bash
streamlit run app.py
```

The application supports:
- soil image upload,
- soil classification,
- nutrient-aware crop recommendation,
- and multimodal agricultural inference.

---

# Running Research Experiments

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
agrosense_v2.ipynb
```

---

# Experimental Focus

Current experimentation includes:
- soil image classification,
- multimodal feature fusion,
- geospatial representation learning,
- agricultural deep learning,
- crop recommendation systems,
- and intelligent agricultural inference.

---

# Research Contributions

AgroSense v2 contributes toward next-generation agricultural AI through:

- multimodal agricultural intelligence,
- integration of geospatial representations,
- end-to-end agricultural inference pipelines,
- scalable experimentation infrastructure,
- and deployable AI-assisted farming systems.

---

# Technology Stack

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

## Research Tooling
- Jupyter Notebook
- Kaggle
- Git LFS

---

# Future Directions

Planned future extensions include:
- satellite imagery integration,
- transformer-based agricultural foundation models,
- explainable agricultural AI,
- temporal agricultural forecasting,
- real-time IoT sensor fusion,
- and distributed training pipelines.

---

# Research References

## Original Research Paper
https://arxiv.org/abs/2509.01344

## Original AgroSense Notebook
https://www.kaggle.com/code/tmleyncodes/research-work-agrosense-agricultural-ai

## AgroSense v2 Notebook
https://www.kaggle.com/code/rishabhhhme/agrosense-2-0

---

# Contributions

Contributions, research collaborations, and feature proposals are welcome.

To contribute:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

---

# License

This project follows the original repository license.

---

# Authors

## Original AgroSense Author

Vishal Pandey  
Email: pandeyvishal.mlprof@gmail.com

---

## AgroSense v2 Contributor & Maintainer

Rishav Tewari

AI/ML Research Engineer  
Agricultural AI • Geospatial Intelligence • Deep Learning

GitHub:  
https://github.com/rishabhh-me

Email:  
rishavtewari.research@gmail.com

---

# Contact

For research discussions, collaborations, or project inquiries:

Vishal Pandey  
pandeyvishal.mlprof@gmail.com

Rishav Tewari  
rishavtewari.research@gmail.com
