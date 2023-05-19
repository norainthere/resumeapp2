import torch
from torch.utils.data import DataLoader
from eqmodel import EQModel
from dataset import EQDataset

# Load the test dataset
test_dataset = EQDataset("test_data.csv")  # Replace with your test dataset

# Create a data loader for the test dataset
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = EQModel()
model.load_state_dict(torch.load("eq_model_weights.pth"))
model.eval()

# Evaluation loop
total_loss = 0.0
total_samples = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

# Calculate average loss
average_loss = total_loss / total_samples
print(f"Average Loss: {average_loss:.4f}")
