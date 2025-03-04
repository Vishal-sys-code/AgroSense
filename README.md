# AgroSense: Integrated Crop Recommendation System

AgroSense is an end-to-end deep learning system for precision agriculture that integrates soil image analysis with quantitative nutrient profiling to provide real-time crop recommendations. Leveraging state-of-the-art convolutional neural networks for soil classification and machine learning models for crop prediction, AgroSense offers a comprehensive decision-support tool for farmers and agronomists.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Experiments and Results](#experiments-and-results)
- [Contributions](#contributions)
- [License](#license)
- [Contact](#contact)

## Overview

Agriculture faces a pressing need to enhance crop yields while sustaining environmental health. Traditional soil testing methods are often slow and expensive, hindering rapid decision-making. AgroSense addresses these challenges by fusing image-based soil classification with real-time nutrient profiling. Our system uses advanced deep learning models to classify soil types from images and integrates these predictions with sensor-based nutrient data to generate optimal crop recommendations.

## Features
- **Soil Image Classification:** Uses multiple deep learning architectures (Custom CNN, ResNet18, EfficientNet-B0, Vision Transformer) to accurately classify soil types from images.
- **Nutrient Profiling:** Incorporates key soil parameters including pH, nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, and rainfall.
- **Data Fusion Pipeline:** Fuses one-hot encoded soil type data with normalized numerical features to form a robust feature vector for crop recommendation.
- **Crop Recommendation:** Predicts the most suitable crop using a multi-layer perceptron (MLP) and benchmarks it against ensemble methods (XGBoost, LightGBM, TabNet).
- **Real-time Web Application:** Deployed as a Streamlit app for seamless user interaction and real-time decision support.

## System Architecture

AgroSense is organized into two main modules:
1. **Soil Classification Module:**  
   - **Input:** Digital soil images.  
   - **Preprocessing:** Images are resized to 224×224 pixels, normalized using ImageNet statistics, and augmented for robustness.  
   - **Modeling:** Advanced CNN architectures extract hierarchical features to classify soil into distinct types.  
   - **Output:** The model produces an integer label, which is then one-hot encoded.
2. **Crop Recommendation Module:**  
   - **Input:** Normalized soil parameters (pH, N, P, K, temperature, humidity, rainfall) combined with the one-hot encoded soil type.  
   - **Fusion:** A data integration pipeline concatenates the visual and numerical data into a unified feature vector.  
   - **Modeling:** A multi-layer perceptron (MLP) predicts crop recommendations, validated against ensemble methods for robust performance.  
   - **Output:** The system delivers a crop recommendation in real-time.

A high-level block diagram and detailed flowcharts are provided within the paper to illustrate the data fusion and decision-making processes.

## Installation

### Prerequisites

- Python 3.8 or higher
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [Torchvision](https://pytorch.org/vision/)
- [NumPy, Pandas, Matplotlib, Seaborn, scikit-learn](https://scikit-learn.org/)
- Other dependencies are listed in `requirements.txt`.

### Steps

1. **Clone the Repository:**
     ```bash
     git clone https://github.com/yourusername/AgroSense.git
     cd AgroSense
     ```
2. **Create and Activate a Virtual Environment (Recommended):**
   ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Unix or MacOS:
    source venv/bin/activate
   ```
3. **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```

# Usage

## **Running the Streamlit Web App**
From the project root, execute the following command to launch the web application:
```bash
streamlit run app.py
```
The app allows users to upload a soil image and input numerical soil parameters. The system then processes the image using deep learning models, fuses the output with normalized nutrient data, and displays the predicted soil type along with the recommended crop.

## Running Experiments
To reproduce and review experimental results, open and run the provided Jupyter Notebook original_notebook.py, which includes detailed performance analyses and visualizations of both the soil classification and crop recommendation modules.


# **Directory Structure**
```txt
AgroSense/
├── Saved Models/
│   ├── crop_recommendation_lgb_model.pkl
│   ├── crop_recommendation_mlp_model.pth
│   ├── crop_recommendation_tabnet_model.pkl
│   ├── crop_recommendation_xgb_model.pkl
│   ├── custom_cnn_model.pth
│   ├── efficientnet_b0_model.pth
│   ├── label_encoder.pkl
│   ├── prediction_pipeline.py
│   ├── resnet18_model.pth
│   ├── scaler.pkl
│   ├── soil_classes.pkl
│   └── vit_model.pth
├── app.py
├── original_notebook.py
├── requirements.txt
└── README.md
```

# Experiments and Results
Our experimental results demonstrate state-of-the-art performance for both the soil classification and crop recommendation modules. Soil classification models, including ResNet18 and EfficientNet-B0, achieved test accuracies of up to 100%, while the integrated crop recommendation module attained accuracies in the range of 97-98%. Comparative visualizations (see Figures in the paper) indicate that our integrated approach outperforms traditional tabular-data-based systems by effectively fusing visual and quantitative soil features.

# Contributions
AgroSense sets a new benchmark for precision agriculture by:
- Integrating soil image analysis with nutrient profiling into a unified, real-time decision support system.
- Demonstrating high accuracy through advanced deep learning models and robust data fusion techniques.
- Providing a scalable, user-friendly Streamlit application for practical agricultural deployment.

# Contact
For further information or collaboration inquiries, please contact:<br>
Vishal Pandey<br>
Email: pandeyvishal.mlprof@gmail.com
