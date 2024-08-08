import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import torchvision.models as models
from collections import defaultdict

# Dataset class for loading ECG images
class ECGDataset(Dataset):
    def __init__(self, base_dir, classes, transform=None, num_leads=12, max_samples_per_class=2000):
        self.base_dir = base_dir
        self.classes = classes
        self.transform = transform if transform else transforms.ToTensor()
        self.num_leads = num_leads
        self.max_samples_per_class = max_samples_per_class
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.base_dir, cls)
            all_samples = []
            for sample_dir in os.listdir(cls_dir):
                sample_path = os.path.join(cls_dir, sample_dir)
                for segment_dir in os.listdir(sample_path):
                    segment_path = os.path.join(sample_path, segment_dir)
                    # Collect all lead images
                    image_files = sorted([f for f in os.listdir(segment_path) if f.endswith('.png')])
                    if len(image_files) >= self.num_leads:
                        all_samples.append(segment_path)
                        if len(all_samples) >= self.max_samples_per_class:
                            break
                if len(all_samples) >= self.max_samples_per_class:
                    break
            samples.extend([(path, cls) for path in all_samples])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dir_path, label = self.samples[idx]
        label_idx = self.classes.index(label)
        
        images = []
        image_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.png')])
        
        # Load images
        for img_file in image_files[:self.num_leads]:  # Use only the first num_leads images
            img_path = os.path.join(dir_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Use cv2 for faster image loading
            
            if img is not None:
                img = Image.fromarray(img)
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        
        # Pad or truncate images to ensure we have exactly self.num_leads images
        if len(images) < self.num_leads:
            size = (224, 224)
            empty_img = Image.new('L', size, color=0)
            for _ in range(self.num_leads - len(images)):
                images.append(self.transform(empty_img))
        elif len(images) > self.num_leads:
            images = images[:self.num_leads]
        
        # Stack images to form (num_leads, H, W) tensor
        images = torch.stack(images, dim=0)  # [num_leads, H, W]
        
        # Remove the single channel dimension if it exists
        if images.shape[1] == 1:
            images = images.squeeze(1)  # Remove the channel dimension if it's of size 1
        
        return images, label_idx

# CNN model for ECG classification
class ECGCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, padding=1)  # 12 input channels
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate

        self.fc1 = nn.Linear(64 * 28 * 28, 1024)  # Corrected size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)  # Output number of classes

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # Output shape [batch_size, 16, 112, 112]
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # Output shape [batch_size, 32, 56, 56]
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # Output shape [batch_size, 64, 28, 28]
        x = x.view(-1, 64 * 28 * 28)  # Flatten for fully connected layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first fully connected layer
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output raw scores (logits)
        return x

# ResNet18 model for ECG classification
class ResNet18ECG(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18ECG, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        # Modify the first convolutional layer to accept 12 input channels
        self.resnet.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final fully connected layer to output the correct number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        model.eval()
        corrects = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", unit="batch"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                corrects += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_accuracy = corrects / total
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        scheduler.step()

    # After the final epoch, calculate evaluation metrics and plot confusion matrix
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Final Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Final Epoch')
    plt.show()
    
    # Save the model
    torch.save(model.state_dict(), 'ecg_cnn_model.pth')
    print("Model saved as 'ecg_cnn_model.pth'")

if __name__ == "__main__":
    base_dir = 'PTB-XL CWT datas'
    classes = ['AMI', 'NORM', 'IMI', 'LMI']
    
    dataset = ECGDataset(base_dir, classes, transform=transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor()  # Convert PIL Image to tensor
    ]), num_leads=12, max_samples_per_class=2000)

    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18ECG(num_classes=len(classes)).to(device)
    print(model)

    train_model(model, train_loader, val_loader, num_epochs=30)
