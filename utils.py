import os
import argparse
import numpy as np
import pandas as pd
import random
import torch
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from model import ConcreteAttentionModel  # 使用你的模型

# 设置随机种子
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ------------------------
# 自定义批处理函数
# ------------------------
def collate(batch):
    """将一个 batch 内的各样本打包成张量，供 DataLoader 使用。"""
    img_features = torch.stack([item[0] for item in batch], dim=0)  
    text_cont = torch.stack([item[1] for item in batch], dim=0)
    labels = torch.stack([item[2] for item in batch])  # 使用 stack 保持每个标签为一个独立元素
    slide_ids = [item[3] for item in batch]
    # print("Batch sizes:", img_features.shape, text_cont.shape, labels.shape)
    return [img_features, text_cont, labels, slide_ids]

# ------------------------
# 自定义数据集类
# ------------------------
class FeatureBagsDataset(Dataset):
    """
    每行文本数据对应一个图片特征文件，以 slide_id 作为拼接文件名的依据。
    """
    def __init__(self, feature_dir, slide_df,
                 continuous_columns, target_column,
                 file_prefix="", file_suffix="_features.npz"):
        """
        参数：
          - feature_dir: 存放图片特征文件的根目录
          - slide_df: 训练或验证阶段对应的 DataFrame，每行至少包含:
              * slide_id 列，用于构造图片特征文件名
              * categorical_columns + continuous_columns 列，用于模型
              * target_column 列，标签
          - file_prefix, file_suffix: 构造图片特征文件名的前后缀。
        """
        self.feature_dir = feature_dir
        self.slide_df = slide_df.reset_index(drop=True)
        self.continuous_columns = continuous_columns
        self.target_column = target_column
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix

    def __len__(self):
        return len(self.slide_df)

    def __getitem__(self, idx):
        row = self.slide_df.iloc[idx]

        # 强制把 slide_id 转成 int（如果本来是浮点数，如 3.0，则变成 3）
        slide_id_raw = row["slide_id"]
        slide_id_int = int(slide_id_raw)  # 若 CSV 中本就有整型，这行无影响

        file_name = f"{self.file_prefix}{slide_id_int}{self.file_suffix}"
        file_path = os.path.join(self.feature_dir, file_name)
        
        # 加载对应的图片特征文件
        # 注意检查: .npz文件中必须有 'features' 键
        data = np.load(file_path)
        img_feature = torch.tensor(data['features'], dtype=torch.float32)

        # 仅提取连续特征（无离散特征）
        continuous_features = torch.tensor(
            row[self.continuous_columns].values, dtype=torch.float32
        )
        
        label = torch.tensor(float(row[self.target_column]), dtype=torch.float32)

        return img_feature, continuous_features, label, slide_id_int

# ------------------------
# 定义数据加载方式
# ------------------------
def define_data_sampling(batch_size, train_dataset, val_dataset, method, workers):
    g = torch.Generator()
    g.manual_seed(0)

    if method == "random":
        print("Random sampling setting")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,  # 每次处理一个样本
            shuffle=True,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        raise Exception(f"Sampling method '{method}' not implemented.")

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader

# ------------------------
# 早停机制
# ------------------------
class MonitorBestModelEarlyStopping:
    def __init__(self, patience=15, min_epochs=20, saving_checkpoint=True):
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.early_stop = False
        self.best_loss = np.Inf
        self.saving_checkpoint = saving_checkpoint

    def __call__(self, epoch, eval_loss, model, log_dir):
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.counter = 0
            print(f'Epoch {epoch} validation loss decreased to {eval_loss:.6f}')
            if self.saving_checkpoint:
                self.save_checkpoint(model, log_dir, epoch)
        else:
            self.counter += 1
            print(f'Early stopping counter {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True

    def save_checkpoint(self, model, log_dir, epoch):
        os.makedirs(log_dir, exist_ok=True)
        # 删除所有旧的 checkpoint 文件
        ckpt_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith("_checkpoint.pt")]
        for file in ckpt_files:
            os.remove(file)
            print(f"Removed old checkpoint: {file}")
        
        # 保存最新的 checkpoint
        filepath = os.path.join(log_dir, f"{epoch}_checkpoint.pt")
        torch.save(model.state_dict(), filepath, _use_new_zipfile_serialization=False)
        print(f"Saved checkpoint: {filepath}")

# ------------------------
# 评估指标等辅助函数
# ------------------------
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def print_model(model):
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} trainable parameters")

def compute_regression_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return mse, mae, r2

def orthogonal_loss(emb1, emb2, alpha_l1=0.01, alpha_l2=0.01):
    """
    emb1, emb2: shape (batch_size, dim)
    同时包含 L1 和 L2 惩罚，让 emb1 · emb2 尽量接近 0
    若只想用 L2，可将 alpha_l1=0; 只想用 L1，可将 alpha_l2=0
    """
    # 点积: (B,)
    dot = torch.sum(emb1 * emb2, dim=-1)
    # L1 = mean(|dot|)
    l1_val = torch.mean(torch.abs(dot))
    # L2 = mean(dot^2)
    l2_val = torch.mean(dot ** 2)

    return alpha_l1 * l1_val + alpha_l2 * l2_val

# ------------------------
# 文本数据预处理
# ------------------------
def preprocess_and_split(data_path, continuous_columns,
                         target_column, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)

    # 如果 csv 没有 "slide_id" 这一列，用行号创建
    if "slide_id" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "slide_id"}, inplace=True)
        print("slide_id 列已自动添加。")

    # 先从连续变量列表中剔除 slide_id
    norm_columns = [col for col in continuous_columns if col != "slide_id"]

    # 对连续变量（待归一化的部分）填充缺失值
    for col in norm_columns:
        df[col] = df[col].fillna(df[col].median())

    # 对连续值归一化操作
    scaler = MinMaxScaler()
    df[norm_columns] = scaler.fit_transform(df[norm_columns])
    # 特征仅为连续变量（无离散特征）
    feature_columns = continuous_columns
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, df

# ------------------------
# 模型训练函数
# ------------------------
def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, log_dir, device, start_epoch=1):
    early_stopping = MonitorBestModelEarlyStopping(patience=5000, min_epochs=5000, saving_checkpoint=True)

    # 定义 alpha_l1, alpha_l2 用于 L1, L2 惩罚，使两个特征尽量保持正交学习更多互补的特征
    alpha_l1 = 0.01
    alpha_l2 = 0.01

    # 用于保存 MSE, MAE, R² 的值
    mse_values = []
    mae_values = []
    r2_values = []
    train_loss_values = []
    val_loss_values = []
    epoch_num = start_epoch + num_epochs - 1  # 默认最后一个epoch

    # 用于记录最佳R²及对应的训练信息
    best_r2 = -float('inf')
    best_epoch = start_epoch
    best_train_loss = None
    best_val_loss = None
    best_mse = None
    best_mae = None

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            img_features, cont_features, labels, _ = batch

            # 将数据迁移到GPU上
            img_features = img_features.to(device)
            cont_features = cont_features.to(device)
            labels = labels.to(device)

            # 仅传入图片和连续特征
            predictions, emb_img, emb_txt = model(img_features, cont_features, return_emb=True)
            loss = loss_fn(predictions.squeeze(), labels)

            # # 计算正交损失，叠加到总loss中
            # loss_orth = orthogonal_loss(emb_img, emb_txt, alpha_l1=alpha_l1, alpha_l2=alpha_l2)
            # loss = loss + loss_orth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_predictions, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                img_features, cont_features, labels, _ = batch

                # 将数据迁移到GPU上
                img_features = img_features.to(device)
                cont_features = cont_features.to(device)
                labels = labels.to(device)

                preds = model(img_features, cont_features)
                loss_v = loss_fn(preds.squeeze(), labels)
                val_loss += loss_v.item()
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        mse, mae, r2 = compute_regression_metrics(all_predictions, all_targets)

        # 存储每个epoch的MSE、MAE、R²
        mse_values.append(mse)
        mae_values.append(mae)
        r2_values.append(r2)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)

        print(f"Epoch {epoch}/{start_epoch + num_epochs - 1}, "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # 更新最佳指标：以R²为标准
        if r2 > best_r2:
            best_r2 = r2
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_mse = mse
            best_mae = mae

        # 早停检查
        early_stopping(epoch, val_loss, model, log_dir)
        if early_stopping.early_stop:
            epoch_num = epoch
            print("Early stopping triggered.")
            break
    
    # 绘图时，x轴使用从 start_epoch 到 epoch_num（包括epoch_num）的范围
    x_axis = list(range(start_epoch, epoch_num + 1))

    # 训练结束后绘制指标
    plt.figure(figsize=(10, 6))

    # 绘制 MSE 曲线
    plt.plot(x_axis, mse_values, label='MSE', color='red')
    # 绘制 MAE 曲线
    plt.plot(x_axis, mae_values, label='MAE', color='blue')

    # 添加标题和标签
    plt.title('Training Process Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Values')

    # 设置横坐标刻度和分割线，每隔 100 画一条
    ticks = np.arange(min(x_axis), max(x_axis) + 1, 100)
    plt.xticks(ticks)

    # 手动设置刻度标签：只有刻度值为 500 的倍数时显示数字
    labels = [str(tick) if tick % 500 == 0 else '' for tick in ticks]
    plt.gca().set_xticklabels(labels)

    plt.legend()

    # 保存图像
    plt.grid(True)
    plt.savefig("Training_Process_Metrics.png")
    plt.close()

    # 训练结束后绘制 R² 指标
    plt.figure(figsize=(10, 6))

    # 绘制 R² 曲线
    plt.plot(x_axis, r2_values, label='R²', color='green')

    # 添加标题和标签
    plt.title('Coefficient of Determination (R²)')
    plt.xlabel('Epochs')
    plt.ylabel('R² Values')

    # 设置横坐标刻度和分割线，每隔 100 画一条
    ticks = np.arange(min(x_axis), max(x_axis) + 1, 100)
    plt.xticks(ticks)

    # 手动设置刻度标签：只有刻度值为 500 的倍数时显示数字
    labels = [str(tick) if tick % 500 == 0 else '' for tick in ticks]
    plt.gca().set_xticklabels(labels)

    plt.legend()

    # 保存图像
    plt.grid(True)
    plt.savefig("Coefficient_of_Determination(R²).png")
    plt.close()

    # 绘制 Train Loss 和 Val Loss
    plt.figure(figsize=(10, 6))
    
    # 绘制训练损失曲线
    plt.plot(x_axis, train_loss_values, label='Train Loss', color='orange')
    # 绘制验证损失曲线
    plt.plot(x_axis, val_loss_values, label='Val Loss', color='purple')

    # 添加标题和标签
    plt.title('Train Loss and Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 设置横坐标刻度和分割线，每隔 100 画一条
    ticks = np.arange(min(x_axis), max(x_axis) + 1, 100)
    plt.xticks(ticks)

    # 手动设置刻度标签：只有刻度值为 500 的倍数时显示数字
    labels = [str(tick) if tick % 500 == 0 else '' for tick in ticks]
    plt.gca().set_xticklabels(labels)

    plt.legend()

    # 显示图像
    plt.grid(True)
    plt.savefig("Train_Val_Loss.png")
    plt.close()

    # 在训练结束后，输出最佳R²对应的Epoch信息
    print(f"训练结束后最佳R²出现在Epoch {best_epoch}/{epoch_num}, "
          f"Train Loss: {best_train_loss:.6f}, Val Loss: {best_val_loss:.6f}, "
          f"MSE: {best_mse:.6f}, MAE: {best_mae:.6f}, R²: {best_r2:.6f}")

# ------------------------
# 参数解析
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="混凝土性能预测模型训练")
    parser.add_argument('--data_path', type=str, required=True, help="文本数据文件路径")
    # parser.add_argument('--categorical_columns', nargs='*', required=[], help="离散值字段列表")
    parser.add_argument('--continuous_columns', nargs='+', required=True, help="连续值字段列表")
    parser.add_argument('--target_column', type=str, required=True, help="目标值字段")
    parser.add_argument('--test_size', type=float, default=0.2, help="测试集比例")
    parser.add_argument('--random_state', type=int, default=42, help="随机种子")
    parser.add_argument('--batch_size', type=int, default=32, help="训练批次大小")
    parser.add_argument('--num_workers', type=int, default=4, help="数据加载器的线程数")
    parser.add_argument('--num_epochs', type=int, default=100, help="训练的总轮数")
    parser.add_argument('--log_dir', type=str, default="checkpoints", help="模型保存路径")
    parser.add_argument('--feature_dir', type=str, required=True, help="图片特征文件所在目录")
    return parser.parse_args()


def load_checkpoint(model, log_dir):
    """
    从 log_dir 中查找 checkpoint 文件，加载最新的 checkpoint。
    返回加载的 epoch，若没有找到 checkpoint，则返回 0（表示从头开始训练）。
    """
    # 查找所有符合命名规则的 checkpoint 文件
    ckpt_files = glob.glob(os.path.join(log_dir, "*_checkpoint.pt"))
    if not ckpt_files:
        print("未找到 checkpoint，重新从头开始训练。")
        return 0

    # 根据文件名解析出 epoch 数，并选择最新的一个（这里假设文件名为 "{epoch}_checkpoint.pt"）
    def get_epoch(fp):
        basename = os.path.basename(fp)
        try:
            # 假设文件名为 "12_checkpoint.pt"
            epoch_str = basename.split('_')[0]
            return int(epoch_str)
        except ValueError:
            return -1

    ckpt_files.sort(key=lambda x: get_epoch(x), reverse=True)
    latest_ckpt = ckpt_files[0]
    start_epoch = get_epoch(latest_ckpt)
    state_dict = torch.load(latest_ckpt)
    model.load_state_dict(state_dict)
    print(f"从 {latest_ckpt} 加载模型，恢复到 epoch {start_epoch}")
    return start_epoch


# ------------------------
# 主程序
# ------------------------
if __name__ == "__main__":
    args = parse_args()
    
    # 定义设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 文本数据预处理和划分
    X_train, X_test, y_train, y_test, df_original = preprocess_and_split(
        data_path=args.data_path,
        continuous_columns=args.continuous_columns,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # 2) 组建 train_df / test_df，并让它们带有 slide_id
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df["slide_id"] = df_original["slide_id"].loc[X_train.index].values

    test_df = pd.concat([X_test, y_test], axis=1)
    test_df["slide_id"] = df_original["slide_id"].loc[X_test.index].values

    # 3) 构建 Dataset
    train_dataset = FeatureBagsDataset(
        feature_dir=args.feature_dir,
        slide_df=train_df,
        continuous_columns=args.continuous_columns,
        target_column=args.target_column
    )

    val_dataset = FeatureBagsDataset(
        feature_dir=args.feature_dir,
        slide_df=test_df,
        continuous_columns=args.continuous_columns,
        target_column=args.target_column
    )

    # 4) 构建 DataLoader
    train_loader, val_loader = define_data_sampling(
        args.batch_size, train_dataset, val_dataset, method="random", workers=args.num_workers
    )
    
    # 5) 初始化模型
    model = ConcreteAttentionModel(
        image_feature_size=524288,
        text_feature_dim=len(args.continuous_columns),
        feature_size_comp=512,
        feature_size_attn=256,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
        fusion_type='kron',
        use_bilinear=True,
        gate_hist=True,
        gate_text=True,
    ).to(device)

    # 6) 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # 7) 尝试加载 checkpoint，获取上次保存的 epoch（如果存在）
    if os.path.exists(args.log_dir):
        last_epoch = load_checkpoint(model, args.log_dir)
    else:
        last_epoch = 0

    # 8) 开始训练
    train_model(
        model, 
        train_loader, 
        val_loader,
        optimizer, 
        loss_fn,
        num_epochs=args.num_epochs,
        log_dir=args.log_dir,
        device=device,
        start_epoch=last_epoch + 1  # 续训从上次 epoch 的下一个 epoch 开始
    )
    
    print("successful")