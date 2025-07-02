import os
import torch
from tqdm import tqdm
from model import DenoisingCNN
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
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
        image = Image.open(img_name).convert('L')   #使用PIL打开图像

        if self.transform:
            image = self.transform(image)

        return image

def denormalize(tensor):  #进行反归一化
    """将张量转换回 [0, 255] 范围的 NumPy 图像，并调整为[512,512,1]形状"""
    result = tensor.numpy()
    result = (result*0.5+0.5)*255   # 将张量转换为numpy数组
    result = np.squeeze(result, axis=0)  # 移除通道维度，形状变为[512,512]
    result = np.expand_dims(result, axis=-1)  # 添加最后一个维度，形状变为[512,512,1]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def test_model(model, dataloader, device, output_dir,ground_truth_dir):
    """测试模型并对测试图像去噪"""
    os.makedirs(output_dir, exist_ok=True)
    ground_truth_list = [f for f in os.listdir(ground_truth_dir) if f.endswith(('.jpg'))]
    ground_img = np.array(Image.open(os.path.join(ground_truth_dir, ground_truth_list[0])))
    with torch.no_grad():  # 不需要计算梯度
        for i, inputs in enumerate(tqdm(dataloader,desc="Testing",total=len(dataloader))):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # 转换为 NumPy 并保存结果
            input_tensor = inputs[0].cpu()
            output_tensor = outputs[0].cpu()
            
            # 如果输入是单通道，添加通道维度
            if input_tensor.shape[0] == 1:
                input_array = input_tensor.numpy().squeeze(0)
                input_array = np.stack([input_array]*3, axis=-1)  # 复制单通道为三通道
            else:
                input_array = input_tensor.numpy().transpose(1, 2, 0)
                
            # 对输出做同样的处理
            if output_tensor.shape[0] == 1:
                output_array = output_tensor.numpy().squeeze(0)
                output_array = np.stack([output_array]*3, axis=-1)
            else:
                output_array = output_tensor.numpy().transpose(1, 2, 0)
                
            # 反归一化并保存
            # 使用denormalize函数将数据从[-1,1]恢复到[0,255]
            input_img = denormalize(inputs[0].cpu())
            output_img = denormalize(outputs[0].cpu())
            input_img = input_img.squeeze(-1)
            output_img = output_img.squeeze(-1)
            ssim_value = ssim(output_img, ground_img,data_range=255)
            psnr_value = psnr(output_img, ground_img,data_range=255)
            print(f" SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}")
            
            # 保存输入图像和去噪后的输出图像
            Image.fromarray(input_img, mode='L').save(os.path.join(output_dir, f'input_{i+1}.png'))
            Image.fromarray(output_img, mode='L').save(os.path.join(output_dir, f'output_{i+1}.png'))

    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    # 配置参数
    data_dir = 'data/test/gray'  # 测试图像所在目录
    model_path = 'denoising_cnn_10epoch.pth'  # 模型权重路径
    output_dir = 'data/test_result'  # 结果保存目录
    ground_truth_dir = 'data/test/gray'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),     #将 PIL 图像转换为张量（[0, 255] → [0, 1]）
        transforms.Normalize((0.5,), (0.5,))  #将张量归一化到 [-1, 1]，单通道
    ])

    # 创建数据集和数据加载器
    dataset = DenoisingDataset(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  #shuffle控制是否随机打乱

    # 加载模型
    model = load_model(model_path, device)

    # 执行测试并保存结果
    test_model(model, dataloader, device, output_dir,ground_truth_dir)