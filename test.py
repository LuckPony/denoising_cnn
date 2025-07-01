import os
import torch
from model import DenoisingCNN
from torchvision import transforms
from PIL import Image
import numpy as np

def load_model(model_path, device):
    """加载训练好的模型"""
    model = DenoisingCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model

class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')   #使用PIL打开图像

        if self.transform:
            image = self.transform(image)

        return image

def denormalize(tensor):  #进行反归一化
    """将张量转换回 [0, 255] 范围的 NumPy 图像"""
    result = tensor.numpy().transpose(1, 2, 0)   #改变通道顺序，从 pytorch格式的(C, H, W) 转换为 numpy格式的(H, W, C)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

def test_model(model, dataloader, device, output_dir):
    """测试模型并对测试图像去噪"""
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():  # 不需要计算梯度
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # 转换为 NumPy 并保存结果
            input_img = denormalize(inputs[0].cpu())   #将输入图像从 [-1, 1] 范围恢复到 [0, 255]
            output_img = denormalize(outputs[0].cpu())

            Image.fromarray(input_img).save(os.path.join(output_dir, f'input_{i}.png'))
            Image.fromarray(output_img).save(os.path.join(output_dir, f'output_{i}.png'))

    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    # 配置参数
    data_dir = 'data/test'  # 测试图像所在目录
    model_path = 'denoising_cnn_50epoch.pth'  # 模型权重路径
    output_dir = 'data/test_result'  # 结果保存目录
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),     #将 PIL 图像转换为张量（[0, 255] → [0, 1]）
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #将张量归一化到 [-1, 1]
    ])

    # 创建数据集和数据加载器
    dataset = DenoisingDataset(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = load_model(model_path, device)

    # 执行测试并保存结果
    test_model(model, dataloader, device, output_dir)