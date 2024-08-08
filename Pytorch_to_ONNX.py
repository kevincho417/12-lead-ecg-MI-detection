import torch
import torch.onnx
from CWT_CNN import CombinedModel
from CWT_CNN import CWTConvNet
from CWT_CNN import ResNet18ECG


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reconstruct the model components
cwt_model = CWTConvNet()
resnet_model = ResNet18ECG(num_classes=4)
combined_model = CombinedModel(cwt_model, resnet_model).to(device)

# Load the state dictionary
combined_model.load_state_dict(torch.load("trained_combined_model.pth"))
combined_model.eval()  # Set the model to evaluation mode

# Define dummy input matching the model's input dimensions
dummy_input = torch.randn(1, 12, 72).to(device)  # Adjust size if needed

# Export the model to ONNX
onnx_file_path = "combined_model.onnx"
torch.onnx.export(
    combined_model,
    dummy_input,
    onnx_file_path,
    verbose=True,
    input_names=['input'],
    output_names=['output'],
    export_params=True,
    opset_version=12  # Use the latest compatible opset version
)

print(f"Model successfully saved to {onnx_file_path}")
