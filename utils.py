import os
import argparse
import numpy as np
import pandas as pd
import random
import torch
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
    text_cat = torch.stack([item[2] for item in batch], dim=0)      
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    slide_ids = [item[4] for item in batch]                         
    return [img_features, text_cont, text_cat, labels, slide_ids]

# ------------------------
# 自定义数据集类
# ------------------------
class FeatureBagsDataset(Dataset):
    """
    每行文本数据对应一个图片特征文件，以 slide_id 作为拼接文件名的依据。
    """
    def __init__(self, feature_dir, slide_df,
                 categorical_columns, continuous_columns, target_column,
                 file_prefix="genview_features_", file_suffix=".npz"):
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
        self.categorical_columns = categorical_columns
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

        # 提取离散和连续特征
        continuous_features = torch.tensor(
            row[self.continuous_columns].values, dtype=torch.float32
        )
        categorical_features = torch.tensor(
            row[self.categorical_columns].values.astype(int), dtype=torch.long
        )
        label = torch.tensor(row[self.target_column], dtype=torch.float32)

        return img_feature, continuous_features, categorical_features, label, slide_id_int

# ------------------------
# 定义数据加载方式
# ------------------------
def define_data_sampling(train_dataset, val_dataset, method, workers):
    g = torch.Generator()
    g.manual_seed(0)

    if method == "random":
        print("Random sampling setting")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1,  # 每次处理一个样本
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
        batch_size=1,
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
        filepath = os.path.join(log_dir, f"{epoch}_checkpoint.pt")
        torch.save(model.state_dict(), filepath)

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

# ------------------------
# 文本数据预处理
# ------------------------
def preprocess_and_split(data_path, categorical_columns, continuous_columns,
                         target_column, test_size=0.2, random_state=42):
    """
    读取CSV并进行:
     1) 若无 slide_id 列则创建(用行号做ID)；若有但为浮点型可稍后再转。
     2) 对离散特征做 LabelEncoder，对连续特征做 MinMaxScaler。
     3) 划分训练/测试集，返回 X_train, X_test, y_train, y_test, label_encoders, 以及带 slide_id 的 df。
    """
    df = pd.read_csv(data_path)

    # 如果 csv 没有 "slide_id" 这一列，可以用行号创建：
    if "slide_id" not in df.columns:
        df.reset_index(inplace=True)  # 将旧索引变成一列 'index'
        df.rename(columns={"index": "slide_id"}, inplace=True)

    # 对连续特征做缺失值填充
    for col in continuous_columns:
        df[col] = df[col].fillna(df[col].median())

    # 对离散特征做缺失值填充+编码
    label_encoders = {}
    for col in categorical_columns:
        df[col] = df[col].fillna("缺失")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 连续特征归一化
    scaler = MinMaxScaler()
    df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

    # 划分特征和标签 (不包含 slide_id)
    feature_columns = categorical_columns + continuous_columns
    X = df[feature_columns]
    y = df[target_column]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, label_encoders, df

# ------------------------
# 模型训练函数
# ------------------------
def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, log_dir):
    early_stopping = MonitorBestModelEarlyStopping(patience=15, min_epochs=20, saving_checkpoint=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            img_features, cont_features, cat_features, labels, _ = batch

            predictions = model(img_features, cont_features, cat_features)
            loss = loss_fn(predictions.squeeze(), labels)

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
                img_features, cont_features, cat_features, labels, _ = batch
                preds = model(img_features, cont_features, cat_features)
                loss_v = loss_fn(preds.squeeze(), labels)
                val_loss += loss_v.item()
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        mse, mae, r2 = compute_regression_metrics(all_predictions, all_targets)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")

        # 早停检查
        early_stopping(epoch, val_loss, model, log_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

# ------------------------
# 参数解析
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="混凝土性能预测模型训练")
    parser.add_argument('--data_path', type=str, required=True, help="文本数据文件路径")
    parser.add_argument('--categorical_columns', nargs='+', required=True, help="离散值字段列表")
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

# ------------------------
# 主程序
# ------------------------
if __name__ == "__main__":
    args = parse_args()

    # 1) 文本数据预处理和划分
    (X_train, X_test,
     y_train, y_test,
     label_encoders, df_original) = preprocess_and_split(
         data_path=args.data_path,
         categorical_columns=args.categorical_columns,
         continuous_columns=args.continuous_columns,
         target_column=args.target_column,
         test_size=args.test_size,
         random_state=args.random_state,
    )

    # 2) 组建 train_df / test_df，并让它们带有 slide_id
    train_df = pd.concat([X_train, y_train], axis=1)
    # slide_id 对应 X_train.index
    train_df["slide_id"] = df_original["slide_id"].loc[X_train.index].values

    test_df = pd.concat([X_test, y_test], axis=1)
    test_df["slide_id"] = df_original["slide_id"].loc[X_test.index].values

    # 3) 构建 Dataset
    train_dataset = FeatureBagsDataset(
        feature_dir=args.feature_dir,
        slide_df=train_df,
        categorical_columns=args.categorical_columns,
        continuous_columns=args.continuous_columns,
        target_column=args.target_column
    )

    val_dataset = FeatureBagsDataset(
        feature_dir=args.feature_dir,
        slide_df=test_df,
        categorical_columns=args.categorical_columns,
        continuous_columns=args.continuous_columns,
        target_column=args.target_column
    )

    # 4) 构建 DataLoader
    train_loader, val_loader = define_data_sampling(
        train_dataset, val_dataset, method="random", workers=args.num_workers
    )

    # 5) 计算离散特征的类别维度
    categorical_dims = [
        len(label_encoders[col].classes_)
        for col in args.categorical_columns
    ]

    # 6) 初始化模型 (根据你的实际设计调整)
    model = ConcreteAttentionModel(
        image_feature_size=524288,
        text_feature_dim=len(args.continuous_columns),
        categorical_dims=categorical_dims,
        embedding_dim=128,
        feature_size_comp=512,
        feature_size_attn=256,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
        fusion_type='kron',
        use_bilinear=True,
        gate_hist=True,
        gate_text=True,
    )

    # 7) 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # 8) 开始训练
    train_model(
        model, train_loader, val_loader,
        optimizer, loss_fn,
        num_epochs=args.num_epochs,
        log_dir=args.log_dir
    )
