import argparse
import os
import numpy as np
import cv2
from PIL import Image
from skimage import io, exposure
import tifffile
from shapely.geometry import Polygon
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from encoders import load_encoder_resnet  # 使用自定义编码器

# 参数解析
def get_args_parser():
    parser = argparse.ArgumentParser('混凝土微观特征提取', add_help=False)
    parser.add_argument("--input_image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--tile_size", type=int, default=512, help="分块尺寸（像素）")
    parser.add_argument("--overlap", type=int, default=128, help="分块重叠像素")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_type", choices=['resnet', 'custom'], default='resnet')
    parser.add_argument("--checkpoint", type=str, help="模型权重路径")
    return parser

# 图像预处理
class GrayImagePreprocessor:
    def __init__(self, clahe_clip=2.0, grid_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, 
                                   tileGridSize=grid_size)
    
    def __call__(self, img_array):
        # 对比度受限直方图均衡化
        enhanced = self.clahe.apply(img_array)
        # 高斯去噪
        denoised = cv2.GaussianBlur(enhanced, (3,3), 0)
        return denoised

# 关键区域检测
def detect_porosity(img, min_area=50):
    """检测孔隙结构"""
    # 自适应阈值
    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 7
    )
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # 轮廓检测
    contours, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤小区域
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# 分块生成
def generate_tiles(img_shape, tile_size, overlap):
    """
    生成分块坐标 (xmin, ymin, xmax, ymax)
    """
    h, w = img_shape
    tiles = []
    
    x_steps = range(0, w, tile_size - overlap)
    y_steps = range(0, h, tile_size - overlap)
    
    for y in y_steps:
        for x in x_steps:
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tiles.append((x, y, x_end, y_end))
    return tiles

# 数据集类
class MicrostructureDataset(Dataset):
    def __init__(self, img_array, tiles, transform=None):
        self.img = img_array
        self.tiles = tiles
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.tiles[idx]
        tile = self.img[y1:y2, x1:x2]
        
        # 转换为伪RGB
        tile_rgb = np.stack([tile]*3, axis=-1)
        
        if self.transform:
            tile_rgb = self.transform(Image.fromarray(tile_rgb))
        
        return tile_rgb, (x1, y1, x2, y2)

# 特征提取流程
def extract_features(args):
    # 初始化
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 读取图像
    if args.input_image.endswith(('.tif', '.tiff')):
        img = tifffile.imread(args.input_image)
    else:
        img = io.imread(args.input_image)
    orig_h, orig_w = img.shape
    
    # 2. 预处理
    preprocessor = GrayImagePreprocessor(clahe_clip=1.5)
    processed_img = preprocessor(img)
    
    # 3. 孔隙检测
    porosity_contours = detect_porosity(processed_img)
    
    # 4. 生成分块
    tiles = generate_tiles(img.shape, args.tile_size, args.overlap)
    
    # 5. 加载模型
    if args.model_type == 'resnet':
        model = load_encoder_resnet('resnet50', args.checkpoint)
    else:
        raise NotImplementedError("自定义模型暂未实现")
    model.to(device)
    model.eval()
    
    # 6. 特征提取
    dataset = MicrostructureDataset(
        processed_img, tiles,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    
    features = []
    coords = []
    
    with torch.no_grad():
        for batch, coord in loader:
            batch = batch.to(device)
            feat = model(batch).cpu().numpy()
            features.append(feat)
            coords.append(coord.numpy())
    
    features = np.concatenate(features, axis=0)
    coords = np.concatenate(coords, axis=0)
    
    # 7. 保存结果
    np.savez(
        os.path.join(args.output_dir, "features.npz"),
        features=features,
        coords=coords,
        porosity=porosity_contours,
        img_shape=(orig_h, orig_w)
    )
    
    print(f"特征已保存至 {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('主程序', parents=[get_args_parser()])
    args = parser.parse_args()
    extract_features(args)