#created by salma filali



import json
from torch.utils.data import Dataset
import torch
import logging

# dataset class to load data from JSON
class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        # Load the dataset from JSON
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['input'], item['label']  # Adjust this according to your JSON format


# Evaluation function to evaluate the model
def evaluate_model(model, test_data_loader, device):
    logging.info("Starting evaluation...")
    model.eval()  # Setting model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f"Evaluation completed. Accuracy: {accuracy:.2f}%")
    return accuracy