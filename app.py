import streamlit as st
import numpy as np
import torch
from PIL import Image
import pickle
from torchvision import transforms
import sys
import io
import os
import torch.nn as nn
from collections import Counter

# Append "Saved Models" directory to sys.path if needed.
sys.path.append("Saved Models")

# ---------------------------
# Define Model Architectures (Outside Cached Functions)
# ---------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.fc1   = nn.Linear(64 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CropMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CropMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

# ---------------------------
# Load Models and Objects (using st.cache_resource)
# ---------------------------
@st.cache_resource
def load_objects():
    # Set device and force map_location to CPU if CUDA is unavailable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = torch.device("cpu") if not torch.cuda.is_available() else device

    cwd = os.getcwd()
    base_dir = os.path.join(cwd, "Saved Models")
    
    # Load scaler, label encoder, and soil classes.
    with open(os.path.join(base_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(base_dir, "soil_classes.pkl"), "rb") as f:
        soil_classes = pickle.load(f)
    
    # Initialize soil classification model (CustomCNN)
    soil_model = CustomCNN(num_classes=len(soil_classes))
    soil_model.load_state_dict(torch.load(os.path.join(base_dir, "custom_cnn_model.pth"), map_location=map_loc))
    soil_model.to(device)
    
    # Initialize crop recommendation model (CropMLP)
    # The saved model expects 7 numerical features plus one-hot encoded soil type.
    crop_input_dim = 7 + len(soil_classes)
    crop_model = CropMLP(input_dim=crop_input_dim, hidden_dim=64, output_dim=len(le.classes_))
    crop_model.load_state_dict(torch.load(os.path.join(base_dir, "crop_recommendation_mlp_model.pth"), map_location=map_loc))
    crop_model.to(device)
    
    # Define image transforms (should match training transforms)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return device, scaler, le, soil_classes, soil_model, crop_model, test_transforms

device, scaler, le, soil_classes, soil_model, crop_model, test_transforms = load_objects()

# ---------------------------
# Helper Functions for Prediction
# ---------------------------
def predict_soil_type(image, model, transform, device):
    model.eval()
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def one_hot_encode(label, num_classes):
    return np.eye(num_classes)[label]

def rule_based_crop_recommendation(soil_numerical_values, soil_label):
    """
    Rule-based crop recommendation using only nutrient values and pH.
    Expected decision order: [N, P, K, pH] extracted from the full 7-parameter array.
    """
    decision_values = soil_numerical_values[:, [0, 1, 2, 5]]  # N, P, K, pH
    N = decision_values[0, 0]
    P = decision_values[0, 1]
    K = decision_values[0, 2]
    pH = decision_values[0, 3]
    
    if pH < 4.0 or pH > 7.5:
        return ["Not suitable for any crop"]
    if N < 50 or P < 20 or K < 20:
        return ["Nutrients deficient - not suitable for any crop"]
    
    soil_type = soil_classes[soil_label]
    recommendations = []
    if soil_type == "Alluvial soil":
        recommendations = ["Rice"]
        if N > 100 and P > 50 and K > 50:
            recommendations.append("Sugarcane")
    elif soil_type == "Black Soil":
        recommendations = ["Maize"]
        if N > 100 and P > 50 and K > 50:
            recommendations.append("Cotton")
    elif soil_type == "Clay soil":
        recommendations = ["Wheat"]
        if N > 100 and P > 50 and K > 50:
            recommendations.append("Barley")
    elif soil_type == "Red soil":
        recommendations = ["Vegetables"]
        if N > 100 and P > 50 and K > 50:
            recommendations.append("Pulses")
    else:
        recommendations = ["Crop recommendation unclear"]
    return recommendations

def integrated_crop_recommendation(soil_image, soil_numerical_values):
    """
    Integrated pipeline:
      1. Predicts soil type from the uploaded soil image.
      2. One-hot encodes the predicted soil type.
      3. Scales the full 7 numerical inputs (order: [N, P, K, Temperature, Humidity, pH, Rainfall]).
      4. Concatenates the scaled numerical features with the one-hot encoded soil type.
      5. Uses rule-based logic (on only [N, P, K, pH]) to get crop recommendations.
      
    Returns:
      - Predicted soil type (string)
      - Recommended crop(s) (list)
      - Final input vector (numpy array)
    """
    soil_label = predict_soil_type(soil_image, soil_model, test_transforms, device)
    predicted_soil = soil_classes[soil_label]
    st.write("Predicted Soil Type:", predicted_soil)
    
    soil_one_hot = one_hot_encode(soil_label, len(soil_classes)).reshape(1, -1)
    
    numerical_scaled = scaler.transform(soil_numerical_values)
    
    final_input = np.concatenate([numerical_scaled, soil_one_hot], axis=1).astype(np.float32)
    st.write("Final Input Vector Shape:", final_input.shape)
    
    recommended_crops = rule_based_crop_recommendation(soil_numerical_values, soil_label)
    st.write("Rule-Based Recommended Crops:", recommended_crops)
    
    return predicted_soil, recommended_crops, final_input

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("AgroSense Crop Recommendation System")
st.write("Upload a soil image and enter soil parameters to get a crop recommendation.")
st.write("Please enter 7 parameters: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature (°C), Humidity (%), pH, Rainfall (mm).")
st.write("Note: For crop decision logic, only nutrients (N, P, K) and pH are used.")

uploaded_file = st.file_uploader("Choose a soil image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_data = uploaded_file.read()
    soil_image = Image.open(io.BytesIO(image_data))
    st.image(soil_image, caption="Uploaded Soil Image", use_column_width=True)
else:
    st.warning("Please upload a soil image.")

st.write("Enter soil parameters:")
N = st.number_input("Nitrogen (N)", value=90)
P = st.number_input("Phosphorus (P)", value=40)
K = st.number_input("Potassium (K)", value=40)
temperature = st.number_input("Temperature (°C)", value=20)
humidity = st.number_input("Humidity (%)", value=80)
pH = st.number_input("pH", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=200)

if st.button("Recommend Crop"):
    if uploaded_file is None:
        st.error("A soil image is required for crop recommendation!")
    else:
        numerical_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
        soil_type, crop_rec, final_input = integrated_crop_recommendation(soil_image, numerical_values)
        st.write("Final Predicted Soil Type:", soil_type)
        st.success("Recommended Crop(s): " + ", ".join(crop_rec))
        st.write("Final Input Vector:", final_input)
