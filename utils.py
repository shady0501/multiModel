import os
import argparse
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from model import ConcreteAttentionModel  # 使用新的模型

# 设置随机种子以确保可重复性
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 数据加载和批处理
def collate(batch):
    img = torch.stack([item[0] for item in batch], dim=0)  # 图像特征
    text_cont = torch.stack([item[1] for item in batch], dim=0)  # 连续特征
    text_cat = torch.stack([item[2] for item in batch], dim=0)  # 离散特征
    label = torch.tensor([item[3] for item in batch], dtype=torch.float32)  # 标签
    slide_id = [item[4] for item in batch]  # 样本ID
    return [img, text_cont, text_cat, label, slide_id]

# 数据集类
class FeatureBagsDataset(Dataset):
    def __init__(self, npz_file_path, slide_df, categorical_columns, continuous_columns, target_column):
        data = np.load(npz_file_path)
        self.features = torch.tensor(data['features'], dtype=torch.float32)  # 图像特征
        self.slide_df = slide_df.reset_index(drop=True)
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.target_column = target_column

    def __len__(self):
        return len(self.slide_df)

    def __getitem__(self, idx):
        feature = self.features[idx]
        categorical_features = torch.tensor(self.slide_df.loc[idx, self.categorical_columns].values, dtype=torch.long)  # 离散特征
        continuous_features = torch.tensor(self.slide_df.loc[idx, self.continuous_columns].values, dtype=torch.float32)  # 连续特征
        label = torch.tensor(self.slide_df[self.target_column][idx], dtype=torch.float32)
        slide_id = self.slide_df["slide_id"][idx]
        return feature, continuous_features, categorical_features, label, slide_id

# 定义数据加载器
def define_data_sampling(train_split, val_split, method, workers):
    # 设置随机种子
    g = torch.Generator()
    g.manual_seed(0)

    # 训练数据加载器
    if method == "random":
        print("Random sampling setting")
        train_loader = DataLoader(
            dataset=train_split,
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

    # 验证数据加载器
    val_loader = DataLoader(
        dataset=val_split,
        batch_size=1,  # 每次处理一个样本
        sampler=SequentialSampler(val_split),
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader

# 早停机制
class MonitorBestModelEarlyStopping:
    def __init__(self, patience=15, min_epochs=20, saving_checkpoint=True):
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.early_stop = False
        self.best_loss = np.Inf

    def __call__(self, epoch, eval_loss, model, log_dir):
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.counter = 0
            print(f'Epoch {epoch} validation loss decreased to {eval_loss:.6f}')
            self.save_checkpoint(model, log_dir, epoch)
        else:
            self.counter += 1
            print(f'Early stopping counter {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True

    def save_checkpoint(self, model, log_dir, epoch):
        filepath = os.path.join(log_dir, f"{epoch}_checkpoint.pt")
        torch.save(model.state_dict(), filepath)

# 获取优化器的学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

# 打印模型结构
def print_model(model):
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")

# 计算回归任务的评估指标
def compute_regression_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return mse, mae, r2

# 数据预处理函数
def preprocess_and_split(data_path, categorical_columns, continuous_columns, target_column, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)

    # 处理缺失值
    for col in continuous_columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_columns:
        df[col].fillna("缺失", inplace=True)

    # 独热编码离散值
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_categorical = ohe.fit_transform(df[categorical_columns])

    ohe_feature_names = ohe.get_feature_names_out(categorical_columns)
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=ohe_feature_names, index=df.index)

    df = pd.concat([df, encoded_categorical_df], axis=1)
    df.drop(columns=categorical_columns, inplace=True)

    # 归一化连续值
    scaler = MinMaxScaler()
    df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

    # 划分训练集和测试集
    feature_columns = list(encoded_categorical_df.columns) + continuous_columns

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# 训练函数
def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, log_dir):
    early_stopping = MonitorBestModelEarlyStopping(patience=15, min_epochs=20, saving_checkpoint=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            img_features, text_cont_features, text_cat_features, labels, _ = batch
            predictions = model(img_features, text_cont_features, text_cat_features)
            loss = loss_fn(predictions.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_predictions, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                img_features, text_cont_features, text_cat_features, labels, _ = batch
                predictions = model(img_features, text_cont_features, text_cat_features)
                val_loss += loss_fn(predictions.squeeze(), labels).item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)

        mse, mae, r2 = compute_regression_metrics(all_predictions, all_targets)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}')

        early_stopping(epoch, val_loss, model, log_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

# 添加参数解析函数
def parse_args():
    parser = argparse.ArgumentParser(description="混凝土性能预测模型训练")
    parser.add_argument('--data_path', type=str, required=True, help="数据文件路径")
    parser.add_argument('--categorical_columns', nargs='+', required=True, help="离散值字段列表")
    parser.add_argument('--continuous_columns', nargs='+', required=True, help="连续值字段列表")
    parser.add_argument('--target_column', type=str, required=True, help="目标值字段")
    parser.add_argument('--test_size', type=float, default=0.2, help="测试集比例")
    parser.add_argument('--random_state', type=int, default=42, help="随机种子")
    parser.add_argument('--batch_size', type=int, default=32, help="训练批次大小")
    parser.add_argument('--num_workers', type=int, default=4, help="数据加载器的线程数")
    parser.add_argument('--num_epochs', type=int, default=100, help="训练的总轮数")
    parser.add_argument('--log_dir', type=str, default="checkpoints", help="模型保存路径")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    X_train, X_test, y_train, y_test = preprocess_and_split(
        data_path=args.data_path,
        categorical_columns=args.categorical_columns,
        continuous_columns=args.continuous_columns,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_split = FeatureBagsDataset("train_features.npz", train_df, args.categorical_columns, args.continuous_columns, args.target_column)
    val_split = FeatureBagsDataset("val_features.npz", test_df, args.categorical_columns, args.continuous_columns, args.target_column)

    train_loader, val_loader = define_data_sampling(train_split, val_split, method="random", workers=args.num_workers)

    model = ConcreteAttentionModel(
        image_feature_size=1024,
        text_feature_dim=len(args.continuous_columns),
        categorical_dims=[10, 3, 3, 3, 3, 10],
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=args.num_epochs, log_dir=args.log_dir)
