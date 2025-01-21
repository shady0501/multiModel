import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置随机种子以确保可重复性
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 数据加载和批处理
def collate(batch):
    # 将批次数据转换为张量
    img = torch.cat([item[0] for item in batch], dim=0)  # 合并图像特征
    text = torch.cat([item[1] for item in batch], dim=0)  # 合并文本特征
    text_cat = torch.stack([item[2] for item in batch])  # 合并文本分类变量
    label = torch.FloatTensor([item[3] for item in batch])  # 抗压强度标签
    slide_id = [item[4] for item in batch]  # 样本ID
    return [img, text, text_cat, label, slide_id]       

# 数据集类
class FeatureBagsDataset(Dataset):
    def __init__(self, df, data_dir, input_feature_size, input_text_size, input_text_cat_size):
        self.slide_df = df.copy().reset_index(drop=True)
        self.data_dir = data_dir
        self.input_feature_size = input_feature_size
        self.input_text_size = input_text_size
        self.input_text_cat_size = input_text_cat_size  # 文本分类变量的类别数
    
    def _get_feature_path(self, slide_id):
        return os.path.join(self.data_dir, f"{slide_id}_Mergedfeatures.pt")

    def __getitem__(self, idx):
        slide_id = self.slide_df["slide_id"][idx]
        label = self.slide_df["compressive_strength"][idx]  # 假设标签列名为 compressive_strength

        # 加载图像特征
        full_path = self._get_feature_path(slide_id)
        features = torch.load(full_path)
        features_merged = torch.from_numpy(np.array([x[0].mean(0) for x in features]))  # 合并特征

        # 加载文本特征（假设文本特征已预处理好）
        text_features = torch.from_numpy(np.load(os.path.join(self.data_dir, f"{slide_id}_text.npy")))  # 文本特征

        # 加载文本分类变量
        text_cat = self.slide_df["text_category"][idx]  # 假设分类变量列名为 text_category
        text_cat = torch.tensor(text_cat, dtype=torch.long)  # 转换为长整型张量

        return features_merged, text_features, text_cat, label, slide_id

    def __len__(self):
        return len(self.slide_df)

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

# 训练函数
def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, log_dir):
    early_stopping = MonitorBestModelEarlyStopping(patience=15, min_epochs=20, saving_checkpoint=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features_merged, features_flattened, labels, _ = batch
            predictions = model(features_merged, features_flattened)
            loss = loss_fn(predictions, labels)
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
                features_merged, features_flattened, labels, _ = batch
                predictions = model(features_merged, features_flattened)
                val_loss += loss_fn(predictions, labels).item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)

        # 计算评估指标
        mse, mae, r2 = compute_regression_metrics(all_predictions, all_targets)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}')

        # 早停机制
        early_stopping(epoch, val_loss, model, log_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

# 主函数
if __name__ == "__main__":
    # 假设 df 是包含数据的 DataFrame，data_dir 是特征文件路径
    df = pd.read_csv("data.csv")  # 加载数据
    data_dir = "features"  # 特征文件目录

    # 定义数据集
    train_split = FeatureBagsDataset(df[df["split"] == "train"], data_dir, input_feature_size=128)
    val_split = FeatureBagsDataset(df[df["split"] == "val"], data_dir, input_feature_size=128)

    # 定义数据加载器
    train_loader, val_loader = define_data_sampling(train_split, val_split, method="random", workers=4)

    # 定义模型、优化器和损失函数
    model = YourModel()  # 替换为你的模型
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()  # 使用均方误差损失

    # 训练模型
    train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=100, log_dir="checkpoints")
