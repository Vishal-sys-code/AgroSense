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

# Append "Saved Models" directory to sys.path if needed
sys.path.append("Saved Models")

# ---------------------------
# Define Model Architectures (outside cached functions)
# ---------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build base directory for saved models.
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
    soil_model.load_state_dict(torch.load(os.path.join(base_dir, "custom_cnn_model.pth"), map_location=device))
    soil_model.to(device)
    
    # Initialize crop recommendation model (CropMLP)
    # Now we expect 4 numerical features: [N, P, K, pH] plus one-hot encoded soil type.
    crop_input_dim = 4 + len(soil_classes)  # e.g., if 4 soil classes, input dim = 4+4 = 8.
    crop_model = CropMLP(input_dim=crop_input_dim, hidden_dim=64, output_dim=len(le.classes_))
    crop_model.load_state_dict(torch.load(os.path.join(base_dir, "crop_recommendation_mlp_model.pth"), map_location=device))
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
    """
    Given a PIL image, predicts the soil type using the soil classification model.
    Returns an integer label.
    """
    model.eval()
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def one_hot_encode(label, num_classes):
    """
    Converts an integer label to a one-hot encoded vector.
    """
    return np.eye(num_classes)[label]

def rule_based_crop_recommendation(soil_numerical_values, soil_label):
    """
    Rule-based crop recommendation.
    Expected order for soil_numerical_values: [N, P, K, pH]
    """
    # Extract key values: here indices 0: N, 1: P, 2: K, 3: pH.
    N = soil_numerical_values[0, 0]
    P = soil_numerical_values[0, 1]
    K = soil_numerical_values[0, 2]
    pH = soil_numerical_values[0, 3]
    
    # Rule for pH: if too acidic (<4.0) or too alkaline (>7.5), no crop is suitable.
    if pH < 4.0 or pH > 7.5:
        return ["Not suitable for any crop"]
    
    # Rule for nutrient deficiency: if any key nutrient is too low.
    if N < 50 or P < 20 or K < 20:
        return ["Nutrients deficient - not suitable for any crop"]
    
    # Get soil type name from soil_classes (assumed to be loaded)
    soil_type = soil_classes[soil_label]
    recommendations = []
    
    # For demonstration, add multiple recommendations if nutrient levels are very high.
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
      3. Scales the numerical inputs (order: [N, P, K, pH]).
      4. Concatenates the scaled numerical features with the one-hot encoded soil type.
      5. Feeds the final input into the crop recommendation model.
      
    Returns the predicted soil type (string), recommended crop(s) (list), and final input vector.
    """
    # Step 1: Predict soil type from image.
    soil_label = predict_soil_type(soil_image, soil_model, test_transforms, device)
    predicted_soil = soil_classes[soil_label]
    st.write("Predicted Soil Type:", predicted_soil)
    
    # Step 2: One-hot encode soil type.
    soil_one_hot = one_hot_encode(soil_label, len(soil_classes)).reshape(1, -1)
    
    # Step 3: Scale numerical features (expects a numpy array of shape (1,4))
    numerical_scaled = scaler.transform(soil_numerical_values)
    
    # Step 4: Concatenate scaled numerical features with one-hot encoded soil type.
    final_input = np.concatenate([numerical_scaled, soil_one_hot], axis=1).astype(np.float32)
    st.write("Final Input Vector Shape:", final_input.shape)
    
    # Step 5: Use rule-based crop recommendation logic (for demonstration)
    recommended_crops = rule_based_crop_recommendation(soil_numerical_values, soil_label)
    st.write("Rule-Based Recommended Crops:", recommended_crops)
    
    return predicted_soil, recommended_crops, final_input

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("AgroSense Crop Recommendation System")
st.write("Upload a soil image and enter soil nutrient values (N, P, K) and pH to get a crop recommendation.")

# Soil image upload
uploaded_file = st.file_uploader("Choose a soil image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_data = uploaded_file.read()
    soil_image = Image.open(io.BytesIO(image_data))
    st.image(soil_image, caption="Uploaded Soil Image", use_column_width=True)
else:
    st.warning("Please upload a soil image.")

# Input fields for soil parameters: Only Nutrients (N, P, K) and pH.
st.write("Enter soil parameters (N, P, K, pH):")
N = st.number_input("Nitrogen (N)", value=90)
P = st.number_input("Phosphorus (P)", value=40)
K = st.number_input("Potassium (K)", value=40)
pH = st.number_input("pH", value=6.5)

if st.button("Recommend Crop"):
    if uploaded_file is None:
        st.error("A soil image is required for crop recommendation!")
    else:
        # Prepare numerical features in the order: [N, P, K, pH]
        numerical_values = np.array([[N, P, K, pH]])
        soil_type, crop_rec, final_input = integrated_crop_recommendation(soil_image, numerical_values)
        st.write("Predicted Soil Type:", soil_type)
        st.success("Recommended Crop(s): " + ", ".join(crop_rec))
        st.write("Final Input Vector:", final_input)