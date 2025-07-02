from torch.utils.data import Dataset
import torch
import numpy as np
class SyntheticDenoisingDataset(Dataset):
    def __init__(self, size=100, img_size=(1, 512, 512)):#实现了继承Dataset类的两个方法
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create a clean image
        clean_image = np.random.rand(*self.img_size).astype(np.float32)
        
        # Add some noise
        noisy_image = clean_image + np.random.normal(0, 0.5, self.img_size).astype(np.float32)
        
        # Clip values to [0, 1]
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return torch.from_numpy(noisy_image), torch.from_numpy(clean_image)