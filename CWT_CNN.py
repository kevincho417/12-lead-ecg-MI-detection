import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
from torchsummary import summary

class ECGDataset(Dataset):
    def __init__(self, data_dir, label, max_samples=None, transform=None):
        """
        Args:
            data_dir (string): Directory with all the .npy files.
            label (int): Class label for the dataset.
            max_samples (int, optional): Maximum number of samples to use from the dataset. If None, use all samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.label = label
        self.transform = transform
        self.max_samples = max_samples
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.data = []
        for file_path in self.file_paths:
            data = np.load(file_path)  # Load the .npy file
            self.data.extend(data)  # Append all segments to the data list
        
        # Apply sample limit if specified
        if self.max_samples is not None:
            self.data = self.data[:self.max_samples]

        # Compute class distribution
        self.class_count = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        segment = self.data[idx]
        segment = torch.tensor(segment, dtype=torch.float32)  # Convert to PyTorch tensor
        
        if self.transform:
            segment = self.transform(segment)

        return segment, self.label

def create_limited_dataloader(data_dirs, labels, max_samples_per_class=2000, batch_size=16):
    datasets = [
        ECGDataset(data_dir=data_dir, label=label, max_samples=max_samples_per_class)
        for data_dir, label in zip(data_dirs, labels)
    ]
    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Define paths and labels
data_dirs = ['PTB-XL datas/AMI', 'PTB-XL datas/NORM', 'PTB-XL datas/IMI', 'PTB-XL datas/LMI']
labels = [1, 0, 2, 3]  # Adjust labels as needed

# Create DataLoader with a limit of 2000 samples per class
combined_dataloader = create_limited_dataloader(data_dirs, labels, max_samples_per_class=2000, batch_size=16)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming CombinedModel and other classes are already defined as provided

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc='Training', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return epoch_loss, epoch_acc, conf_matrix
import torch
import numpy as np
import torchvision.models as models
from torch import nn
import torch.nn.functional as F

class CWTConvNet(nn.Module):
    def __init__(self):
        super(CWTConvNet, self).__init__()
        dt = 0.004
        w = 6
        coeff = np.sqrt(w * w + 2)
        scales = (np.reciprocal(np.arange(40, 4, -36 / 112)) * (coeff + w)) / (4. * np.pi)
        filters = [None] * len(scales)
        for scale_idx, scale in enumerate(scales):
            M = 10 * scale / dt
            t = np.arange((-M + 1) / 2., (M + 1) / 2.) * dt
            if len(t) % 2 == 0:
                t = t[0:-1]
            norm = (dt / scale) ** .5
            x = t / scale
            wavelet = np.exp(1j * w * x)
            wavelet -= np.exp(-0.5 * (w ** 2))
            wavelet *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)
            filters[scale_idx] = norm * wavelet

        self._cuda = torch.cuda.is_available()
        self.set_filters(filters)
        self.img_select = np.linspace(0, 71, 224, dtype=int)

    def set_filters(self, filters, padding_type='SAME'):
        assert isinstance(filters, list)
        assert padding_type in ['SAME', 'VALID']

        self._filters = [None] * len(filters)
        for ind, filt in enumerate(filters):
            assert filt.dtype in (np.float32, np.float64, np.complex64, np.complex128)

            if np.iscomplex(filt).any():
                chn_out = 2
                filt_weights = np.asarray([np.real(filt), np.imag(filt)], np.float32)
            else:
                chn_out = 1
                filt_weights = filt.astype(np.float32)[None, :]

            filt_weights = np.expand_dims(filt_weights, 1)
            filt_size = filt_weights.shape[-1]
            padding = self._get_padding(padding_type, filt_size)

            conv = nn.Conv1d(1, chn_out, kernel_size=filt_size, padding=padding, bias=False)
            conv.weight.data = torch.from_numpy(filt_weights)
            conv.weight.requires_grad_(False)

            if self._cuda:
                conv.cuda()
            self._filters[ind] = conv

    @staticmethod
    def _get_padding(padding_type, kernel_size):
        assert isinstance(kernel_size, int)
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            return (kernel_size - 1) // 2
        return 0

    def forward(self, x):
        batch_size, channels, seq_length = x.shape
        if not self._filters:
            raise ValueError('PyTorch filters not initialized. Please call set_filters() first.')

        results = [None] * len(self._filters)
        for ind, conv in enumerate(self._filters):
            filtered_channels = []
            for channel in range(channels):
                single_channel_input = x[:, channel, :].unsqueeze(1)
                filtered_channel = conv(single_channel_input)
                filtered_channels.append(filtered_channel)
            filtered_channels = torch.stack(filtered_channels, dim=1)
            results[ind] = filtered_channels

        results = torch.stack(results)
        cwt = results.permute(1, 2, 0, 3, 4).contiguous()
        cwt = (cwt[:, :, :, 0, :] + cwt[:, :, :, 1, :] * 1j)
        cwt = cwt[:, :, :224, self.img_select]
        # print(f"CWT Shape before reshape: {cwt.shape}")  # Debug print
        
        cwt = cwt.type(torch.FloatTensor).to(x.device)
        return cwt

class ResNet18ECG(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18ECG, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = ResNet18ECG(num_classes=4).to(device)
resnet_model.load_state_dict(torch.load("ecg_cnn_model.pth"))
resnet_model.eval()

class CombinedModel(nn.Module):
    def __init__(self, cwt_model, resnet_model):
        super(CombinedModel, self).__init__()
        self.cwt_model = cwt_model
        self.resnet_model = resnet_model

    def forward(self, x):
        x = self.cwt_model(x)
        x = self.resnet_model(x)
        return x

cwt_model = CWTConvNet()
combined_model = CombinedModel(cwt_model, resnet_model).to(device)
# Set up the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwt_model = CWTConvNet()
resnet_model = ResNet18ECG(num_classes=4).to(device)
combined_model = CombinedModel(cwt_model, resnet_model).to(device)

# Load the pre-trained ResNet weights
resnet_model.load_state_dict(torch.load("ecg_cnn_model.pth"))
resnet_model.eval()  # Ensure the model is in evaluation mode
combined_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

# Create DataLoader with a limit of 2000 samples per class
data_dirs = ['PTB-XL datas/AMI', 'PTB-XL datas/NORM', 'PTB-XL datas/IMI', 'PTB-XL datas/LMI']
labels = [1, 0, 2, 3]  # Adjust labels as needed
combined_dataloader = create_limited_dataloader(data_dirs, labels, max_samples_per_class=2000, batch_size=16)
if __name__ == "__main__":
    summary(combined_model, (12, 72))

    # Training loop
    num_epochs = 10  # Number of epochs
    for epoch in range(num_epochs):
        train_loss, train_acc = train(combined_model, combined_dataloader, criterion, optimizer, device)
        val_loss, val_acc, val_conf_matrix = evaluate(combined_model, combined_dataloader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Confusion Matrix:\n{val_conf_matrix}")

    # Save the trained model
    torch.save(combined_model.state_dict(), "trained_combined_model.pth")
