# 1. 环境配置

- 虚拟环境配置（仅用于测试代码是否运行成功而非训练使用）：
![alt text](mdImage/env_run.png)

- 环境更新：mmcv 在python 3.8——3.11，3.12支持不好，需要降到3.12以下，因此更换镜像将环境降到3.10：
  ![alt text](mdImage/env_run2.png)

---

# 2. 安装依赖库
```bash
pip install opencv-python
pip install numpy
pip install shapely
pip install scikit-learn
pip install pillow
pip install matplotlib
pip install scikit-image tifffile

pip install -U openmim
mim install mmengine
mim install mmcv>=2.0.0
pip install mmpretrain
```

---

# 3. 代码

### 预训练使用模型
[使用 2024 年的 genview 模型](https://huggingface.co/Xiaojie0903/genview_pretrained_models/tree/main)


### 3.1 预训练文件
- `preprocess.ipynb`
- `extract_features.py`
- `encoders.py`

**图片特征提取的运行命令：**
  *(checkpoints文件和代码处于同一级目录)*
```bash
python extract_features.py \
  --input_image input_images/preprocess_demo.png \
  --checkpoint mocov3_resnet50_8xb512-amp-coslr-100e_in1k_genview.pth \
  --model_type genview_resnet50 \
  --tile_size 512 \
  --overlap 128 \
  --batch_size 32 \
```

### 3.2 训练预测文件
- `utils.py`
- `models.py`

---

# 4. 开发日志

### 2025.1.20
1. preprocess.ipynb 图像预处理调整并运行成功，图像如下：
![alt text](mdImage/image_preprocess.png)

2. 使用 AI 调整完成大部分流程，包括`encoders.py`、`extract_features.py`、`models.py`、`utils.py`

### 2025.1.23
1. 修复 GitHub 提交问题，在保证代码更新不变前提下丢失了近5条推送记录(╥﹏╥)，重新提交
2. 由于原论文中 EsViT 中提供的 checkpoints 无法下载，通过调研使用2024新模型 genview 
3. 嵌入 genview 模型
  - 调整完成模型预训练和图片特征提取部分，使用 genview 模型 ([GitHub 地址](https://github.com/xiaojieli0903/genview/))
  - 模型使用的checkpoints为 Hugging Face 中提供的checkpoints，如图所示：
    ![alt text](mdImage/checkpoints.png)
4. 成功提取灰度图片特征，将其保存为output/genview_features.npz 用于后续训练