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
    parser.add_argument("--input_image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="特征保存路径")
    parser.add_argument("--tile_size", type=int, default=512, help="分块大小")
    parser.add_argument("--overlap", type=int, default=128, help="分块重叠大小")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--model_type", choices=['genview_resnet50', 'genview_vit'], 
                        default='genview_resnet50', help="选择GenView模型类型")
    parser.add_argument("--checkpoint", type=str, required=True, help="预训练权重路径")
    return parser

class MicrostructureDataset(Dataset):
    def __init__(self, img_array, tiles, tile_size, transform=None):
        self.img = img_array
        self.tiles = tiles
        self.tile_size = tile_size

        # 使用常见的RGB图像归一化参数
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
            print("单通道图像")
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

    # 提取 slide_id，从文件名提取 slide_id，例如 "1.png" -> "1"
    slide_id = os.path.basename(args.input_image).split('.')[0]  

    # 检查输入文件是否存在
    if not os.path.isfile(args.input_image):
        raise FileNotFoundError(f"无法找到输入图像文件: {args.input_image}")
    
    # 读取图像（tif/tiff 使用 tifffile，否则使用 skimage.io）
    if args.input_image.endswith(('.tif', '.tiff')):
        img = tifffile.imread(args.input_image).astype(np.uint8)
    else:
        img = io.imread(args.input_image)

    # 判断图像类型：若灰度图（ndim==2或ndim==3且通道数==1），则使用灰度预处理；
    # 若RGB图（ndim==3且通道数==3），则直接使用RGB图像
    if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
        # 灰度图处理：如果是 (H,W,1)，squeeze后得到 (H,W)
        if img.ndim == 3:
            img = np.squeeze(img, axis=-1)
        preprocessor = GrayImagePreprocessor(clahe_clip=1.5)
        processed_img = preprocessor(img)
        # 此时 processed_img 为单通道灰度图
    elif img.ndim == 3 and img.shape[-1] == 3:
        # RGB图像直接使用
        processed_img = img
    else:
        raise ValueError("不支持的图像格式")

    # 生成分块，若 processed_img 为灰度图，则形状为 (H,W)，若RGB则取前两个维度
    if processed_img.ndim == 2:
        tiles = generate_tiles(processed_img.shape, args.tile_size, args.overlap)
    else:
        tiles = generate_tiles(processed_img.shape[:2], args.tile_size, args.overlap)

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
        os.path.join(args.output_dir, f"{slide_id}_features.npz"),
        features=features,
        coords=np.array(coords),  # 确保 coords 是 numpy 数组
        img_shape=img.shape,
        model_type=args.model_type,
        slide_id=slide_id  # 保存 slide_id
    )
    print(f"特征提取完成！结果已保存到: {os.path.join(args.output_dir, f'{slide_id}_features.npz')}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # 如果输入为文件夹，则遍历其中所有有效图像文件
    if os.path.isdir(args.input_image):
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        image_files = [os.path.join(args.input_image, f) for f in os.listdir(args.input_image) if f.lower().endswith(valid_ext)]
        for image_path in image_files:
            args.input_image = image_path
            print(f"正在处理: {image_path}")
            extract_features(args)
    else:
        extract_features(args)
