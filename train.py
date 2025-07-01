import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import DenoisingCNN
from tqdm import tqdm

from preData_synthetic import SyntheticDenoisingDataset



def train(model, dataloader, criterion, optimizer, device):#参数分别代表模型、数据集、损失函数、优化器、设备
    model.train()
    running_loss = 0.0
    
    for inputs, targets in tqdm(dataloader,desc='Training',leave=False): #tqdm用于显示训练进度条
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

if __name__ == '__main__':
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model instance
    model = DenoisingCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss for image reconstruction
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    
    # Create synthetic dataset and data loader
    dataset = SyntheticDenoisingDataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Train for a few epochs
    epochs = 10
    for epoch in range(epochs):
        loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    outdir = f'denoising_cnn_{epochs}epoch.pth'
    # Save model
    torch.save(model.state_dict(), outdir)