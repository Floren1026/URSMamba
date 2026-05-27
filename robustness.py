import torch
import torchvision.transforms.functional as TF
from torchvision.io import encode_jpeg, decode_jpeg

from io import BytesIO
import numpy as np
from PIL import Image

def add_gaussian_noise(image, mean=0.0, std=0.002):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)  # 保证像素值仍在 [0,1] 范围内

def add_multiplicative_noise(image, mean=0.0, std=0.002):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image * (1 + noise)
    return torch.clamp(noisy_image, 0, 1)

def add_salt_and_pepper_noise(image, prob=0.002):
    mask = torch.rand_like(image)  # 生成 [0,1] 之间的随机数
    salt = (mask > (1 - prob)).float()  # 最高 prob 部分变白
    pepper = (mask < prob).float()  # 最低 prob 部分变黑
    noisy_image = torch.clamp(image + salt - pepper, 0, 1)
    return noisy_image

def add_rayleigh_noise(image, scale=0.002):
    """
    给图像添加瑞利噪声

    :param image: 输入图像，形状为 [C, H, W] 或 [B, C, H, W]
    :param scale: 瑞利分布的尺度参数 σ（控制噪声强度）
    :return: 添加噪声后的图像
    """
    noise = scale * torch.sqrt(-2 * torch.log(1 - torch.rand_like(image)))
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)  # 保持像素在 [0,1] 范围内

def add_gamma_noise(image, shape=2.0, scale=0.002):
    """
    给图像添加加性伽马噪声

    :param image: 输入图像，形状为 [C, H, W] 或 [B, C, H, W]
    :param shape: 伽马分布的形状参数 k
    :param scale: 伽马分布的尺度参数 θ（控制噪声强度）
    :return: 添加噪声后的图像
    """
    gamma_dist = torch.distributions.Gamma(concentration=shape, rate=1.0/scale)
    noise = gamma_dist.sample(image.shape).to(image.device)
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)  # 保持像素在 [0,1] 范围内



def center_crop(image: torch.Tensor, remove_ratio=0.5):
    """
    从输入图像 (C, H, W) 中裁剪掉中心区域，保留边缘部分。

    Args:
        image (torch.Tensor): 输入的图像张量，形状为 (C, H, W)。
        remove_ratio (float): 需要移除的中心区域比例（相对于图像大小）。

    Returns:
        torch.Tensor: 中心部分被裁剪掉的图像。
    """
    B, C, H, W = image.shape  # 获取通道数、高度、宽度

    # 计算需要裁剪掉的中心区域大小
    crop_H = int(H * remove_ratio)
    crop_W = int(W * remove_ratio)

    # 计算中心区域的起始位置
    top = (H - crop_H) // 2
    left = (W - crop_W) // 2

    # 复制张量并填充中心区域为 0
    cropped_image = image.clone()
    cropped_image[:, :, top:top + crop_H, left:left + crop_W] = 0  # 黑色填充（0值）

    return cropped_image

def right_crop(image, crop_w, fill_value=0):
    """
    裁剪图像右侧 crop_w 个像素，并用 fill_value 填充。
    
    参数：
    - image: 输入图像 (C, H, W)
    - crop_w: 需要裁剪的宽度
    - fill_value: 用于填充的值（默认 0，即黑色）
    
    返回：
    - 处理后的图像 (C, H, W)
    """
    B, C, H, W = image.shape
    new_image = torch.full((B, C, H, W), fill_value, dtype=image.dtype, device=image.device)  # 创建黑色背景
    new_w = W - crop_w  # 计算裁剪后的宽度
    new_image[:, :, :, :new_w] = image[:, :, :, :new_w]  # 复制左侧部分
    return new_image



def jpeg_compress(img_tensor, quality=90):
    """
    使用 PyTorch 进行 JPEG 编码和解码模拟压缩损失，并将结果转换回 [0,1] 范围。
    :param img_tensor: 形状为 (B, C, H, W) 的张量，值范围 [0, 1] 或 [0, 255]
    :param quality: JPEG 压缩质量 (0-100)，默认 90
    :return: 经过 JPEG 压缩后的张量，值范围 [0, 1]
    """
    # 确保图像是 uint8 格式
    if img_tensor.dtype != torch.uint8:
        img_tensor = (img_tensor * 255).byte()

    img_tensor = img_tensor.cpu()
    # 遍历 batch 进行 JPEG 压缩
    compressed_imgs = []
    for img in img_tensor:
        img_jpeg = encode_jpeg(img, quality=quality)  # 编码 JPEG
        img_decoded = decode_jpeg(img_jpeg)  # 解码 JPEG
        img_decoded = img_decoded.float() / 255.0  # 转换回 [0,1] 浮点数
        compressed_imgs.append(img_decoded)

    return torch.stack(compressed_imgs).cuda()



def rotate(img, angle=45):
    rotated_img = TF.rotate(img, angle, fill=(0, 0, 0))

    return rotated_img