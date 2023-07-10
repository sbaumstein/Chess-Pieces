import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import csv
import os
import cv2

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Step 1: Data Loading and Processing
class WaldoDataset(Dataset):
    def __init__(self, data_folder, csv_file, transform=None):
        self.data = self.load_data(data_folder, csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = self.data[index]['image_path']
        target = self.data[index]['target']
        
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(target, dtype=torch.float32)  # Convert target to a tensor

        return image, target

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
        nn.Linear(32 * 56 * 56, num_classes)  # Output 4 numbers for bounding box coordinates
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
            
            # Reshape the targets to match the output format
            targets = targets.view(-1, 4)
            
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
import cv2
import numpy as np

import cv2
import numpy as np

import cv2

def inference(model, image, test_image_name):
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Pass the image through the model
    with torch.no_grad():
        outputs = model(image)

    # Extract the predicted bounding box coordinates
    predicted_coords = outputs.squeeze().tolist()

    # Load the original image using OpenCV for visualization
    original_image = cv2.imread(test_image_name)

    # Convert the coordinates to pixel values
    img_height, img_width, _ = original_image.shape
    xmin = int(predicted_coords[0] * img_width)
    ymin = int(predicted_coords[1] * img_height)
    xmax = int(predicted_coords[2] * img_width)
    ymax = int(predicted_coords[3] * img_height)

    # Draw the bounding box on the original image
    thickness = 2
    color = (0, 0, 255)  # BGR format: (Blue, Green, Red)
    cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), color, thickness)

    # Display the image with the bounding box
    cv2.imshow('Inference Result', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main script
if __name__ == '__main__':
    # Set your data paths, hyperparameters, and other configurations
    batch_size = 36
    num_epochs = 50
    learning_rate = 0.001
    csv_file = "annotations.csv"
    data_folder = "training_images"

    # Data transformations -- resizes, converts to tensor and normalizes every image so it can be used for deep learning
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and data loaders
    dataset = WaldoDataset(data_folder, csv_file, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model instance
    num_classes = 4  # 4 coordinates for bounding box
    model = create_cnn(num_classes)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Use mean squared error loss for bounding box regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    evaluate_model(model, test_loader)

    # Perform inference

    test_image_name = '1test.jpg'
    test_image = Image.open(test_image_name).convert('RGB')
    inference(model, test_image, test_image_name)
