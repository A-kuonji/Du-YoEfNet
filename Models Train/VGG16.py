# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import os
import time
import copy
import csv  # 新增：CSV记录

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# --------------------------
# 配置参数
# --------------------------
TRAIN_DATA_DIR = 'data1/train'
VAL_DATA_DIR = 'data1/val'
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
MODEL_NAME = "VGG16_data1"
OUTPUT_DIR = 'output'
CSV_PATH = os.path.join(OUTPUT_DIR, 'training_log.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据增强配置
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# 训练一个 epoch
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    return running_loss / total_samples, correct_predictions / total_samples

# 验证一个 epoch
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return (running_loss / total_samples, correct_predictions / total_samples,
            precision, recall, f1, all_labels, all_preds)

# 主流程
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(VAL_DATA_DIR, transform=data_transforms['val'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        *list(model.classifier.children())[:-1],
        nn.Dropout(p=0.5),
        nn.Linear(4096, len(train_dataset.classes))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))

    # 打开 CSV 并写表头
    csv_file = open(CSV_PATH, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'precision', 'recall', 'f1'])

    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, precision, recall, f1, labels, preds = validate_epoch(model, val_loader, criterion, device)

        # 写入 CSV
        csv_writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, precision, recall, f1])
        csv_file.flush()

        # 更新学习率
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{MODEL_NAME}_best.pth")

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | F1: {f1:.4f}")

    csv_file.close()
    print(f"训练结果已保存到 {CSV_PATH}")
