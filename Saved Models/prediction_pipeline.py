
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pickle

def predict_soil_type(image_path, model, transform, device):
    """
    Predicts the soil type from an image using the provided soil classification model.
    Returns an integer label.
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def one_hot_encode(label, num_classes):
    """
    Converts an integer label into a one-hot encoded vector.
    """
    return np.eye(num_classes)[label]

def integrated_crop_recommendation(soil_image_path, soil_numerical_values,
                                   soil_model, crop_model, scaler, le, soil_classes,
                                   transform, device):
    """
    Integrated pipeline that:
      1. Takes a soil image and numerical measurements (order: [N, P, K, temperature, humidity, pH, rainfall]).
      2. Predicts the soil type using the provided soil classification model.
      3. One-hot encodes the predicted soil type.
      4. Scales the numerical features using the provided scaler.
      5. Concatenates the scaled numerical features with the one-hot encoded soil type.
      6. Feeds the final input vector into the crop recommendation model to predict the recommended crop.
      
    Returns the predicted crop as a string.
    """
    # Predict soil type
    soil_label = predict_soil_type(soil_image_path, soil_model, transform, device)
    soil_type_name = soil_classes[soil_label]
    print("Predicted Soil Type:", soil_type_name)
    
    # One-hot encode the soil type
    num_soil_classes = len(soil_classes)
    soil_one_hot = one_hot_encode(soil_label, num_soil_classes).reshape(1, -1)
    
    # Scale the numerical features (assumed to be in raw form, order: [N, P, K, temperature, humidity, pH, rainfall])
    numerical_features_scaled = scaler.transform(soil_numerical_values)
    
    # Concatenate numerical features with one-hot encoded soil type
    final_input = np.concatenate([numerical_features_scaled, soil_one_hot], axis=1).astype(np.float32)
    
    # Predict crop using the crop recommendation model
    final_tensor = torch.tensor(final_input, dtype=torch.float32).to(device)
    crop_model.eval()
    with torch.no_grad():
        crop_output = crop_model(final_tensor)
        _, crop_pred_label = torch.max(crop_output, 1)
    predicted_crop = le.inverse_transform(crop_pred_label.cpu().numpy())
    return predicted_crop[0]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load saved objects
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("soil_classes.pkl", "rb") as f:
        soil_classes = pickle.load(f)
    
    # Define image transforms (should match training transforms)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Define your soil classification model architecture (example using CustomCNN)
    import torch.nn as nn
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
    
    soil_model = CustomCNN(num_classes=len(soil_classes))
    soil_model.load_state_dict(torch.load("custom_cnn_model.pth", map_location=device))
    soil_model.to(device)
    
    # Define your crop recommendation model architecture (example using an MLP)
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
    
    # For example, input_dim=11 (7 numerical + 4 soil one-hot), hidden_dim=64, output_dim = number of crop classes.
    crop_model = CropMLP(input_dim=11, hidden_dim=64, output_dim=len(le.classes_))
    crop_model.load_state_dict(torch.load("crop_recommendation_mlp_model.pth", map_location=device))
    crop_model.to(device)
    
    # Example usage:
    sample_soil_image = "path/to/sample_soil_image.jpg"  # Update with an actual image path
    soil_numerical_values = np.array([[90, 40, 40, 20, 80, 6.5, 200]])
    
    recommended_crop = integrated_crop_recommendation(sample_soil_image, soil_numerical_values,
                                                      soil_model, crop_model, scaler, le, soil_classes,
                                                      test_transforms, device)
    print("Final Crop Recommendation:", recommended_crop)
