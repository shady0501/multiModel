虚拟环境配置（仅用于测试代码是否运行成功而非训练使用）：
![alt text](mdImage/env_run.png.png)

安装依赖库：
pip install opencv-python
pip install numpy
pip install shapely
pip install scikit-learn
pip install pillow
pip install matplotlib

预训练文件：preprocess.ipynb
训练预测文件：utils.py

2025.1.20
1. preprocess.ipynb 图像预处理调整并运行成功，图像如下：
![alt text](mdImage/image_preprocess.png.png)

2. 使用 AI 调整完成大部分流程，包括encoders.py、extract_features.py、models.py、utils.py



# 预训练特征提取步骤：

使用EsViT（Efficient Self-supervised Vision Transformer）进行预训练和特征提取可以分为以下几个主要步骤。以下是一个详细的指南：

---

### **1. 环境准备**
确保你的环境中安装了必要的库：
```bash
pip install torch torchvision timm opencv-python
git clone https://github.com/microsoft/esvit
cd esvit
pip install -r requirements.txt
```

---

### **2. 数据准备**
准备你的自定义数据集，目录结构如下：
```
data/
  train/
    class1/
      img1.jpg
      img2.jpg
      ...
    class2/
      ...
  val/
    ...
```

---

### **3. 自监督预训练**
EsViT的核心优势在于自监督预训练。使用以下脚本进行预训练：

#### **配置文件**
创建配置文件 `config.yaml`：
```yaml
model:
  name: esvit_vit_small
  pretrained: False
  drop_path_rate: 0.1

data:
  dataset: "custom_dataset"
  root: "data/train"
  input_size: 224
  batch_size: 64
  num_workers: 8

training:
  epochs: 100
  optimizer: adamw
  lr: 3e-4
  weight_decay: 0.05
  warmup_epochs: 10

self_sup:
  method: "dino"  # 使用DINO自监督方法
  teacher_temp: 0.07
  student_temp: 0.1
  center_momentum: 0.9
```

#### **启动训练**
```bash
python train_self_supervised.py \
  --cfg config.yaml \
  --output_dir ./output \
  --gpu 0
```

---

### **4. 特征提取**
预训练完成后，使用训练好的模型提取特征：

#### **加载预训练权重**
```python
import torch
from models import build_model

# 加载预训练模型
model = build_model(
    arch="vit_small",
    pretrained_weights="./output/checkpoint.pth",  # 预训练权重路径
    use_self_supervised=True
)
model.eval()
```

#### **提取特征**
```python
from torchvision import transforms
from PIL import Image

# 定义预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 提取单张图像特征
def extract_features(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]
    
    with torch.no_grad():
        features = model(img_tensor)  # 提取CLS token或全局平均池化
    
    return features.squeeze().numpy()  # 转为numpy数组

# 示例
features = extract_features("data/train/class1/img1.jpg", model)
print(f"特征维度: {features.shape}")
```

---

### **5. 下游任务应用**
提取的特征可用于分类、检索等任务：

#### **分类示例（使用Scikit-learn）**
```python
from sklearn.svm import SVC
import numpy as np
import os

# 加载所有特征和标签
X, y = [], []
for class_dir in os.listdir("data/train"):
    for img_path in os.listdir(f"data/train/{class_dir}"):
        features = extract_features(f"data/train/{class_dir}/{img_path}", model)
        X.append(features)
        y.append(class_dir)

# 训练分类器
clf = SVC()
clf.fit(X, y)

# 预测
test_features = extract_features("data/test/img.jpg", model)
pred = clf.predict([test_features])
print(f"预测类别: {pred[0]}")
```

---

### **关键参数说明**
| 参数                | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| `--arch`            | 模型架构（如 `vit_small`, `vit_base`）                               |
| `--pretrained_weights` | 预训练权重路径                                                     |
| `--use_self_supervised` | 是否使用自监督预训练                                             |
| `method: "dino"`    | 自监督方法（DINO、MoCo等）                                           |

---

### **性能优化建议**
1. **多卡训练**：
   ```bash
   torchrun --nproc_per_node=4 train_self_supervised.py --cfg config.yaml
   ```
2. **混合精度训练**：
   ```yaml
   training:
     fp16: True
   ```
3. **特征缓存**：
   ```python
   # 提前提取并保存所有特征
   np.save("train_features.npy", X)
   np.save("train_labels.npy", y)
   ```

---

通过以上步骤，你可以完成EsViT的自监督预训练和特征提取，并将其应用于材料科学图像分析任务。