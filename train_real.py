import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model import DenoisingCNN
from preData_real import get_real_data_loader

def train_model(model,dataloader,criterion, optimizer, model_save_path, epochs=10):
    """训练去噪模型
    
    Args:
        model (nn.Module): 定义的模型
        dataloader (DataLoader): 训练数据加载器，提供数据对
        criterion (nn.Module): 设置的损失函数
        optimizer：训练的优化器设置
        model_save_path (str): 模型保存路径
        epochs (int): 训练轮数
    """
   
    
    # 训练循环
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', total=len(dataloader))
        
        for noisy_inputs, clean_targets in progress_bar:
            # 将输入数据移动到设备
            noisy_inputs = noisy_inputs.to(device)
            clean_targets = clean_targets.to(device)
            
            # 前向传播
            outputs = model(noisy_inputs)
            loss = criterion(outputs, clean_targets)  # 使用干净图像作为目标
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新损失值
            running_loss += loss.item() * noisy_inputs.size(0)
            
            # 更新进度条描述
            progress_bar.set_postfix(loss=loss.item())
        
        # 计算并打印平均损失
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1} Loss: {epoch_loss:.4f}')
        
        # 保存模型检查点
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
    
    print('Training complete')

if __name__ == '__main__':
    # 配置参数
    data_dir = 'data/train/noisy'  # 带噪声的真实图像数据目录
    clean_data_dir = 'data/train/groundTruth'  # 对应的干净图像数据目录
    model_save_path = 'models/denoising_cnn.pth'  # 模型保存路径
    batch_size = 16  # 批处理大小
    shuffle = True  # 是否在每个epoch开始时打乱数据
    epochs = 10  # 训练轮数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择设备
     # 初始化模型并移动到指定设备
    model = DenoisingCNN().to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    # 创建数据加载器
    dataloader = get_real_data_loader(data_dir, clean_data_dir, batch_size, shuffle)
    
    # 开始训练
    train_model(model,dataloader, device,criterion, optimizer, model_save_path, epochs)