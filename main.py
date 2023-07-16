import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Step 1: Data Loading and Processing
class ChessPieceDataset(Dataset):
    def __init__(self, data, folder_path, transform=None):
        self.data = data
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_filename = self.data['filename'][index]
        image_path = os.path.join(self.folder_path, image_filename)
        
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        target_str = self.data['class'][index]  # Use the 'class' column as the target
        target = self.get_target_index(target_str)  # Convert target to class index
        target = torch.tensor(target, dtype=torch.long)  # Convert target to tensor

        return image, target

    def get_target_index(self, target_str):
        class_mapping = {
            'black-king': 0, 'black-queen': 1, 'black-rook': 2, 'black-bishop': 3,
            'black-knight': 4, 'black-pawn': 5, 'white-king': 6, 'white-queen': 7,
            'white-rook': 8, 'white-bishop': 9, 'white-knight': 10, 'white-pawn': 11
        }
        return class_mapping[target_str]

# Step 2: Model Architecture
def create_cnn():
    model = nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(6, 16, 5),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 13 * 13, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 12)
    )
    return model


# Step 3: Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
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
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    print(f'Accuracy: {100 * correct / total}%')


def predict_chess_pieces(model, image_path, transform):
    class_mapping_inverse = {0: 'black-king', 1: 'black-queen', 2: 'black-rook', 3: 'black-bishop', 4: 'black-knight', 5: 'black-pawn', 6: 'white-king', 7: 'white-queen', 8: 'white-rook', 9: 'white-bishop', 10: 'white-knight', 11: 'white-pawn'}
    piece_values = {'king': 0, 'queen': 9, 'rook': 5, 'bishop': 3, 'knight': 3, 'pawn': 1}
    white_score = 0
    black_score = 0

    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_mapping_inverse[predicted.item()]
        color, piece = predicted_class.split('-')

        if color == 'white':
            white_score += piece_values[piece]
        else:
            black_score += piece_values[piece]

    return white_score, black_score

# Main script
if __name__ == '__main__':
    # Load data
    csv_file = 'train_annotations.csv'
    train_data = 'train'
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    annotations = pd.read_csv('train_annotations.csv', names=column_names, skiprows=1)
    annotations.dropna(inplace=True)  # Drop rows with missing values
    class_mapping = {'black-king': 0, 'black-queen': 1, 'black-rook': 2, 'black-bishop': 3, 'black-knight': 4, 'black-pawn': 5, 'white-king': 6, 'white-queen': 7, 'white-rook': 8, 'white-bishop': 9, 'white-knight': 10, 'white-pawn': 11}
    data = [{'filename': row['filename'], 'class': row['class']} for _, row in annotations.iterrows()]
    # Create test datasets and data loaders
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_dataset = ChessPieceDataset(data, train_data, transform=transform)  # Pass the data DataFrame to the dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create model
    model = create_cnn()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # Evaluate the model
    evaluate_model(model, train_loader)

    # After training
    test_image_filename = input('Enter the name of the test image file (e.g., test_image.jpg): ')
    test_image_path = os.path.join('path_to_your_test_images_folder', test_image_filename)
    white_score, black_score = predict_chess_pieces(model, test_image_path, transform)
    print(f'White Score: {white_score}, Black Score: {black_score}')
    if white_score > black_score:
        print('White is up by', white_score - black_score, 'points')
    elif black_score > white_score:
        print('Black is up by', black_score - white_score, 'points')
    else:
        print('The game is tied')