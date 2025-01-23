import argparse
import os
import numpy as np
import cv2
from PIL import Image
from skimage import io
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from mmpretrain import get_model

# 模型名称映射
MODEL_NAME_MAP = {
    'genview_resnet50': 'mocov3_resnet50_8xb512-amp-coslr-100e_in1k',
    'genview_vit': 'mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k'
}

def get_args_parser():
    parser = argparse.ArgumentParser('混凝土微观特征提取', add_help=True)
    parser.add_argument("--input_image", type=str, required=True, help="输入灰度图像路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="特征保存路径")
    parser.add_argument("--tile_size", type=int, default=512, help="分块大小")
    parser.add_argument("--overlap", type=int, default=128, help="分块重叠大小")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--model_type", choices=['genview_resnet50', 'genview_vit'], 
                        default='genview_resnet50', help="选择GenView模型类型")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="预训练权重路径")
    return parser

class MicrostructureDataset(Dataset):
    def __init__(self, img_array, tiles, tile_size, transform=None):
        self.img = img_array
        self.tiles = tiles
        self.tile_size = tile_size

        # 动态设置 Normalize 的 mean 和 std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transform or transforms.Compose([
            transforms.Resize(tile_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        x, y, x_end, y_end = self.tiles[idx]
        tile = self.img[y:y_end, x:x_end]

        # 如果图像是单通道，扩展为 3 通道
        if len(tile.shape) == 2:  # 检查是否为灰度图
            tile = np.stack([tile] * 3, axis=-1)  # 复制单通道到 3 通道

        # 填充到固定大小
        padded_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        h, w = tile.shape[:2]
        padded_tile[:h, :w, :] = tile

        tile = Image.fromarray(padded_tile)
        return self.transform(tile), self.tiles[idx]

def load_genview_model(model_type, checkpoint_path, device):
    """加载GenView预训练模型"""
    if model_type not in MODEL_NAME_MAP:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    model_name = MODEL_NAME_MAP[model_type]
    model = get_model(model=model_name, device=device)

    # 加载预训练权重
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"无法找到指定的权重文件: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model

class GrayImagePreprocessor:
    def __init__(self, clahe_clip=2.0, grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=grid_size)

    def __call__(self, img_array):
        enhanced = self.clahe.apply(img_array)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return denoised

def generate_tiles(img_shape, tile_size, overlap):
    h, w = img_shape
    tiles = []
    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tiles.append((x, y, x_end, y_end))
    return tiles

def extract_features(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查输入文件是否存在
    if not os.path.isfile(args.input_image):
        raise FileNotFoundError(f"无法找到输入图像文件: {args.input_image}")
    
    # 读取图像
    if args.input_image.endswith(('.tif', '.tiff')):
        img = tifffile.imread(args.input_image).astype(np.uint8)
    else:
        img = io.imread(args.input_image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 图像预处理
    preprocessor = GrayImagePreprocessor(clahe_clip=1.5)
    processed_img = preprocessor(img)

    # 生成分块
    tiles = generate_tiles(img.shape, args.tile_size, args.overlap)

    # 加载GenView模型
    model = load_genview_model(args.model_type, args.checkpoint, device)

    # 初始化数据集和 DataLoader
    dataset = MicrostructureDataset(processed_img, tiles, args.tile_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    features = []
    coords = []

    # 提取特征
    with torch.no_grad():
        for batch, coord in loader:
            batch = batch.to(device)
            if 'resnet' in args.model_type:
                feat = model.backbone(batch)[-1].flatten(start_dim=1)
            elif 'vit' in args.model_type:
                feat = model.backbone(batch)[0][:, 0]

            features.append(feat.cpu().numpy())
            coords.extend(coord)  # 修复 numpy 错误，改为 extend

    # 保存特征
    features = np.concatenate(features, axis=0)
    np.savez_compressed(
        os.path.join(args.output_dir, "genview_features.npz"),
        features=features,
        coords=np.array(coords),  # 确保 coords 是 numpy 数组
        img_shape=img.shape,
        model_type=args.model_type
    )
    print(f"特征提取完成！结果已保存到: {os.path.join(args.output_dir, 'genview_features.npz')}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    extract_features(args)
