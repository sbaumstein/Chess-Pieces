import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import csv
import os

from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Step 1: Data Loading and Processing
class WaldoDataset(Dataset):
    def __init__(self, data_folder, csv_file, transform=None):
        self.data = self.load_data(data_folder, csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def load_data(self, data_folder, csv_file):
        data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                image_filename = row[0]
                image_path = os.path.join(data_folder, image_filename)
                target = [int(coord) for coord in row[4:]]  # Assuming coordinates are integers

                data.append({
                    'image_path': image_path,
                    'target': target
                })

        return data

# Step 2: Model Architecture: Using a generic fast Convolutional Neural Network and editing as training goes on
def create_cnn(num_classes):
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 56 * 56, num_classes)
    )
    return model


# Step 3: Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Step 4: Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    # TODO: Implement evaluation code and compute metrics

# Step 5: Inference
def inference(model, image):
    model.eval()
    # TODO: Preprocess the image and pass it through the model
    # TODO: Obtain the predicted positions or class labels

# Step 7: Fine-tuning and Improvements
# TODO: Experiment with different architectures, hyperparameters, and techniques

# Main script
if __name__ == '__main__':
    # Set your data paths, hyperparameters, and other configurations
    data_path = 'path/to/your/dataset'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    csv_file = "/Users/sambaumstein/Desktop/WheresWaldo/Where-s-Waldo-1/annotations.csv"
    data_folder = "/Users/sambaumstein/Desktop/WheresWaldo/Where-s-Waldo-1/training_images"

    # Data transformations -- resizes, converts to tensor and normalizes every image so it can be used for deep learning
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and data loaders
    dataset = WaldoDataset(data_folder, csv_file, transform=transform)
    #CHANGE
    batch_size = len(dataset)
    # Processes data to load
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model instance
   # classes = 2 (Waldo, Not Waldo)
    num_classes = 2
    model = create_cnn(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    evaluate_model(model, test_loader)

    # Perform inference
    test_image = Image.open('path/to/your/test/image.jpg').convert('RGB')
    inference(model, test_image)
