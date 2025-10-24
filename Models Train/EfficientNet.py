# -*- coding: utf-8 -*-
import sys
import io
import os
import time
import csv  # 新增：CSV记录
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# 配置参数
# --------------------------
TRAIN_DATA_DIR = 'data3/train'
VAL_DATA_DIR = 'data3/val'
BATCH_SIZE = 40
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 28
OUTPUT_DIR = 'output/data3'
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
CSV_PATH = os.path.join(LOGS_DIR, 'nopre_epoch_metrics.csv')  # CSV文件路径

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 修复控制台中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Matplotlib 中文配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# EfficientNet 输入尺寸映射
INPUT_SIZE_MAP = {
    'efficientnet_b0': 224, 'efficientnet_b1': 240, 'efficientnet_b2': 260,
    'efficientnet_b3': 300, 'efficientnet_b4': 380, 'efficientnet_b5': 456,
    'efficientnet_b6': 528, 'efficientnet_b7': 600
}
INPUT_SIZE = INPUT_SIZE_MAP[MODEL_NAME.lower()]

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(INPUT_SIZE + 32),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}


class InferenceWrapper(nn.Module):
    """
        模型推理包装器，用于在推理阶段对输入数据进行预处理并应用softmax

        这个包装器类可以:
        1. 对输入进行缩放（例如从0-255缩放到0-1）
        2. 调整通道顺序（例如从channels_last转为channels_first）
        3. 应用归一化（减去均值并除以标准差）
        4. 对模型输出应用softmax获取类别概率分布
        """
    def __init__(self, model: nn.Module,
                 mean: torch.Tensor, std: torch.Tensor,
                 scale_inp: bool = False,
                 channels_last: bool = False):
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.scale_inp = scale_inp
        self.channels_last = channels_last
        self.softmax = nn.Softmax(dim=1)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_inp:
            x = x / 255.0
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.model(x)
        x = self.softmax(x)
        return x

# 创建模型
def create_efficientnet(model_name: str, num_classes: int) -> nn.Module:
    try:
        creator = getattr(models, model_name)
    except AttributeError:
        raise ValueError(f"不支持的EfficientNet版本: {model_name}")
    model = creator(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        nn.Linear(in_features, num_classes)
    )
    return model

# 训练/验证函数
def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        device: torch.device
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if device.type == 'cuda':
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        class_names: list[str]
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return {
        'loss': total_loss / total,
        'acc': correct / total,
        'cm': confusion_matrix(all_labels, all_preds),
        'report': classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    }

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    # 数据加载
    train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(VAL_DATA_DIR, transform=data_transforms['val'])
    CLASS_NAMES = train_dataset.classes
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 模型与优化器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_efficientnet(MODEL_NAME.lower(), NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler() if device.type == 'cuda' else None

    # 打开 CSV 并写入表头
    csv_file = open(CSV_PATH, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device, CLASS_NAMES)

        # 保存历史并写 CSV
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        csv_writer.writerow([epoch+1, train_loss, train_acc, val_metrics['loss'], val_metrics['acc']])
        csv_file.flush()

        # 保存最佳模型
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            torch.save(model, os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_best.pth"))

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.2%}, val_loss: {val_metrics['loss']:.4f}, val_acc: {val_metrics['acc']:.2%}")

    # 关闭 CSV
    csv_file.close()

    # 绘制并保存曲线与混淆矩阵等（略，保持原有逻辑）
    print(f"训练完成，总耗时: {time.time() - start_time:.1f}秒，最佳验证准确率: {best_acc:.2%}")
