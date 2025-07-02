import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RealImageDataset(Dataset):
    """真实图像数据集类，支持灰度图像模式和数据对"""
    def __init__(self, root_dir, clean_root_dir, transform=None):
        """
        Args:
            root_dir (string): 包含带噪声图像的目录路径
            clean_root_dir (string): 包含对应干净图像的目录路径
            transform (callable, optional): 应用于图像的转换操作
        """
        self.root_dir = root_dir
        self.clean_root_dir = clean_root_dir
        self.transform = transform
        # 支持的文件类型：.png, .jpg, .jpeg
        self.image_files = [f for f in os.listdir(root_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        noisy_img_path = os.path.join(self.root_dir, img_name)
        clean_img_path = os.path.join(self.clean_root_dir, img_name)
        
        # 使用PIL库打开图像，转换为灰度图像模式'L'
        noisy_image = Image.open(noisy_img_path).convert('L')  # 带噪声图像
        clean_image = Image.open(clean_img_path).convert('L')   # 干净图像

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image  # 返回数据对

def get_real_data_loader(data_dir, clean_data_dir, batch_size=1, shuffle=False):
    """创建并返回一个处理真实图像的数据加载器"""
    # 数据预处理流程遵循规范：
    # transforms.ToTensor(): 将PIL图像转换为张量（范围从 [0, 255] → [0, 1]）
    # transforms.Normalize((0.5,), (0.5,)): 单通道图像归一化参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 单通道图像归一化参数
    ])

    dataset = RealImageDataset(data_dir, clean_data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, 
                           shuffle=shuffle, num_workers=0)
    
    return data_loader