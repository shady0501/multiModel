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

4. 成功提取灰度图片特征，将其保存为output/genview_features.npz 用于后续训练。
   当前特征维度：{features: (9, 524288), coords: (4, 9), img_shape: (2,), model_type: ()}
  - `features`: (9, 524288) 表明提取的特征矩阵有 9 个分块，每个分块的特征维度为 524288。
  - `coords`: (4, 9) 提供了每个分块的坐标信息（如分块在原始图像中的位置）。
  - `img_shape`: (2,) 描述了输入图像的形状，应该是 (高度, 宽度)。
  - `model_type`: () 表示使用的模型类型（可能在保存时未正确写入）。
  
5. 文本数据预处理（类型及类别数量统计，以便后续处理）

| 字段名称         | 数值类型        | 离散值类别数量          |
|------------------|---------------|-----------------|
| 混凝土强度等级   | 离散值           | 10类    |
| 胶材总量（kg/m³）| 连续值           | -       |
| 水胶比           | 连续值          | -        |
| 水泥品种         | 离散值          | 3类      |
| 水泥型号         | 离散值          | 3类      |
| 水泥（kg/m³）    | 连续值          | -        |
| 粉煤灰掺量（%）  | 连续值          | -        |
| 矿粉掺量（%）    | 连续值          | -        |
| 硅灰掺量（%）    | 连续值          | -        |
| 其余胶材（%）    | 连续值          | -        |
| 减水剂（%）      | 连续值          | -        |
| 砂率             | 连续值         | -        |
| 石子（kg/m³）    | 连续值          | -        |
| 石子级配         | 离散值          | 3类      |
| 抗裂剂（%）      | 连续值          | -        |
| 抗裂剂掺加方式   | 离散值          | 3类      |
| 抗裂剂型号       | 离散值          | 10类     |
| 龄期             | 连续值          | -        |
| 强度             | 连续值          | -        |

### 2025.1.24
1. 更改 `extract_features.py` 逻辑，使其能够支持对多张图片进行特征提取，提取特征保存为 `output/{slide_id}_features.npz`，并重命名测试图片为 `1.png`

2. 更改`utils.py`文件，使其能够符合混凝土的输入格式，当前训练预测的命令为：
  
  ```bash
  python utils.py \
      --data_path data.csv \
      --categorical_columns "混凝土强度等级" "水泥品种" "水泥型号" "石子级配" "抗裂剂掺加方式" "抗裂剂型号" \
      --continuous_columns "胶材总量（kg/m³）" "水胶比" "水泥（kg/m³）" "粉煤灰掺量（%）" "矿粉掺量（%）" "硅灰掺量（%）" "其余胶材（%）" "减水剂（%）" "砂率" "石子（kg/m³）" "抗裂剂（%）" "龄期" \
      --target_column "强度" \
      --test_size 0.2 \
      --random_state 42 \
      --batch_size 32 \
      --num_workers 4 \
      --num_epochs 100 \
      --log_dir checkpoints
  ```
  参数说明：
  - data_path：CSV 数据文件路径。
  - categorical_columns：离散值字段列表，需用空格分隔字段名称。
  - continuous_columns：连续值字段列表，需用空格分隔字段名称。
  - target_column：目标值字段名称。
  - test_size：测试集比例（默认 0.2）。
  - random_state：随机种子（默认 42）。
  - batch_size：训练批次大小（默认 32）。
  - num_workers：数据加载器的线程数（默认 4）。
  - num_epochs：训练的总轮数（默认 100）。
  - log_dir：模型保存路径（默认 checkpoints）。

3. 更改`models.py`文件，使其能够符合混凝土的输入格式，由于此次模型的更改，上述第二条中`utils.py`的代码和运行命令也需要更改，已更改，尚未运行测试
  ```bash
  python utils.py \
    --data_path "data/concrete_data.csv" \
    --categorical_columns "混凝土强度等级" "水泥品种" "水泥型号" "石子级配" "抗裂剂掺加方式" "抗裂剂型号" \
    --continuous_columns "胶材总量（kg/m³）" "水胶比" "水泥（kg/m³）" "粉煤灰掺量（%）" "矿粉掺量（%）" \
    "硅灰掺量（%）" "其余胶材（%）" "减水剂（%）" "砂率" "石子（kg/m³）" "抗裂剂（%）" "龄期" \
    --target_column "强度" \
    --test_size 0.2 \
    --random_state 42 \
    --batch_size 32 \
    --num_workers 4 \
    --num_epochs 100 \
    --log_dir "checkpoints"
  ```
  参数说明：
  - data_path：CSV 数据文件路径。
  - categorical_columns：离散值字段列表，需用空格分隔字段名称。
  - continuous_columns：连续值字段列表，需用空格分隔字段名称。
  - target_column：目标值字段名称。
  - test_size：测试集比例（默认 0.2）。
  - random_state：随机种子（默认 42）。
  - batch_size：训练批次大小（默认 32）。
  - num_workers：数据加载器的线程数（默认 4）。
  - num_epochs：训练的总轮数（默认 100）。
  - log_dir：模型保存路径（默认 checkpoints）。

### 2025.2.3
1. 修改`model.py`的问题：
    - 主模型默认特征维度和图片实际特征维度不一致，应改为**524288**
    - 当前仍然使用**kron**乘积融合，但是可能会导致维度爆炸的问题，可能需要修改为**concat**融合，当前仍然为**kron**乘积融合

### 2025.2.4
1. 将文本数据格式从`xlsx`转换为`csv`，从而匹配代码格式
2. 修改utils.py文件代码，使其能够解析输出的参数命令，并将独热编码改为标签编码（因为在多模态模型中无需使用独热编码防止高维空间距离错误，仅需要通过标签编码嵌入模型）
3. 修改文本数据和图片特征数据的映射关系，每次获取数据时动态加载对应的特征文件
4. 构造包含`slide_id`的假文本数据`data_419.csv`（仅三行），同时对应图片特征均为同一份图片特征拷贝三份重命名得到的
5. 由于原来构造的csv格式的数据不是UTF-8格式，因此需要将`data_419.csv`转换为UTF-8格式，转换后的文件名为`data_419_utf8.csv`
6. 当前存在模型运行计算过程中维度不匹配问题

  运行命令：
  ```bash
  python utils.py \
    --data_path "data_419_utf8.csv" \
    --categorical_columns "Strength of Cement" "Gradation of concrete" "Type of water reducer" "Type of air entrainer" "cement admixture" \
    --continuous_columns "Content of cement (kg/m3)" "Cement/water" "Dosage of FA(%)" "Sand rate" "Dosage of water reducer (%)" "Dosage of air entrainer (%)" "The temperature of concrete(℃)" "The temperature of air(℃)" "Air content(%)" \
    --target_column "The 28day compressive strength of concrete" \
    --test_size 0.2 \
    --random_state 42 \
    --batch_size 32 \
    --num_workers 4 \
    --num_epochs 100 \
    --log_dir "checkpoints" \
    --feature_dir "output"
  ```

### 2025.2.6——2025.2.10
1. 修改模型的线性层的输入维度和实际传入的数据维度不匹配问题
2. 模型运行基本成功（调试使用的服务器内存太小，运行后系统盘爆满）
   ![alt text](mdImage/run_almost_success.png)

### 2025.2.12
查看混凝土微观图像，明确应该识别的图像特征
1. 白色——水泥熟料——关键强度指标——含量与强度负相关
2. 黑色圆形——粉煤灰——当前不考虑
3. 黑色——C-S-H——含量与强度正相关
![alt text](mdImage/microscopic_image.png)

### 2025.2.13——2025.2.14
使用VCCTL软件模拟构造100个混凝土微观图像，见`data/xxx.png`

### 2025.2.15
1. 使用模拟构造的图像数据进行初步实验预测
2. 更换文本数据，因此运行命令改为：
```bash
  python utils.py \
    --data_path "data.csv" \
    --continuous_columns "slide_id" "水用量（kg/m3）" "水泥矿物组分_C3S（%）" "水泥矿物组分_C2S（%）" "水泥矿物组分_C3A（%）" "水泥矿物组分_C4AF（%）" "胶砂抗压_7d（MPa）" "胶砂抗压_28d（MPa）" "胶砂抗折3d（MPa）" "胶砂抗折7d（MPa）" "胶砂抗折_28d（MPa）" "水化热_1d（J/g）" "水化热_3d（J/g）" "水化热_7d（J/g）" "水泥用量（kg/m3）" "FA化学组分_SiO2（%）" "FA化学组分_Al2O3（%）" "FA化学组分_Fe2O3（%）" "FA化学组分_CaO（%）" "粉煤灰用量（kg/m3）" "砂表观密度（kg/m3）" "砂用量（kg/m3）" "石1坚固性（%）" "石1压碎指标（%）" "石1用量（kg/m3）" "石2坚固性（%）" "石2压碎指标（%）" "石2用量（kg/m3）" "石3坚固性（%）" "石3压碎指标（%）" "石3用量（kg/m3）" "外加剂1用量（%）" "外加剂2用量（%）" \
    --target_column "28d抗压强度" \
    --test_size 0.2 \
    --random_state 42 \
    --batch_size 32 \
    --num_workers 4 \
    --num_epochs 100 \
    --log_dir "checkpoints" \
    --feature_dir "output"
```

3. 当前文件夹中文件组织如图所示
   ![alt text](mdImage/fille_organization.png)

4. 文件说明：
   - `data.csv`——混凝土配合比数据，如 "胶砂抗压_28d（MPa）" "胶砂抗折3d（MPa）" 等
   - `data` 文件夹——存放模拟构造的混凝土微观图像
   - `input` 文件夹——存放最初测试特征提取的图片
   - `output` 文件夹——存放最终提取的特征
  
5. 图片特征提取运行命令：
 ```bash
  python extract_features.py \
    --input_image data/1.png \
    --checkpoint mocov3_resnet50_8xb512-amp-coslr-100e_in1k_genview.pth \
    --model_type genview_resnet50 \
    --tile_size 512 \
    --overlap 128 \
    --batch_size 32 \
  ```
6. 更改`extract_features.py`文件的main函数，使其能够自动对文件夹中的所有图像进行特征提取而无需手动指定（当然也可以手动指定）
   更新后的运行命令如下：
```bash
  python extract_features.py \
    --input_image data \
    --checkpoint mocov3_resnet50_8xb512-amp-coslr-100e_in1k_genview.pth \
    --model_type genview_resnet50 \
    --tile_size 512 \
    --overlap 128 \
    --batch_size 32 \
```

7. 由于更新后的文本数据不存在离散值，因此需要对`utils.py`文件和`model.py`文件进行修改，使其能够正确运行
8. 同时由于仅模拟生成了100个图像，`data.csv`文件中数据过多，需要删除部分数据仅保留前100个，而全部数据命名为`data_all.csv`
9. 由于checkpoints太多导致磁盘爆满，因此更新checkpoints的保存原则，改为仅保留10个checkpoints，新来的checkpoints替换掉最旧的checkpoints
10.  由于测试最初结果不是很理想，因此计划使用房屋数据进行测试，检测是否代码原因导致效果不好
    ![alt text](mdImage/first_training.png)

    是否是由于将slide_id也作为训练数据进行训练导致效果不好？

### 2025.2.16
1. 房屋数据集上的运行命令：
     运行命令：
  ```bash
  python utils.py \
    --data_path "houseData.csv" \
    --categorical_columns "卧室数量" "浴室数量" "邮编" \
    --continuous_columns "面积" \
    --target_column "价格" \
    --test_size 0.2 \
    --random_state 42 \
    --batch_size 32 \
    --num_workers 4 \
    --num_epochs 100 \
    --log_dir "checkpoints" \
    --feature_dir "output"
  ```
  当前房屋多模态数据集最佳预测结果：
  ![alt text](mdImage/house_best_prediction.png)

2. 当前混凝土多模态数据集最佳预测结果:
   ![alt text](mdImage/best_prediction1.png)