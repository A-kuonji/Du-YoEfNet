import torch
import timm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib
import torch.optim as optim  # 导入 torch.optim 模块
from tqdm import tqdm  # 导入 tqdm 模块，用于进度条
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from timm.data import resolve_data_config

matplotlib.use('TkAgg')
# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体，支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建输出文件夹
output_dir = "run/data3/output"
os.makedirs(output_dir, exist_ok=True)
images_dir = os.path.join(output_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# 选择模型名称
model_name = 'resnet50'

# 加载模型 (不使用预训练权重)
model = timm.create_model(model_name, pretrained=False)  # 从头开始训练

# 获取数据预处理配置
config = resolve_data_config({}, model=model)  # 使用timm的resolve_data_config

# 创建图像预处理变换
transform = transforms.Compose([
    transforms.Resize(config['input_size'][1:], interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(config['input_size'][1:]),
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std'])
])

if 'vit' in model_name:
    transform = transforms.Compose([
        transforms.Resize(config['input_size'], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

# 设置数据集路径
data_dir = 'data3'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# 创建数据集
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 获取类别数量
num_classes = len(train_dataset.classes)
print(f"类别数量: {num_classes}")


# 微调模型替换最后一层全连接层, 并添加dropout
if hasattr(model, 'fc'):
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),  # 添加 dropout
        torch.nn.Linear(model.fc.in_features, num_classes)
    )
elif hasattr(model, 'classifier'):
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),  # 添加 dropout
        torch.nn.Linear(model.classifier.in_features, num_classes)
    )
elif hasattr(model, 'head'):  # for ViT like models
    model.head = torch.nn.Sequential(
       torch.nn.Dropout(p=0.5),  # 添加 dropout
        torch.nn.Linear(model.head.in_features, num_classes)
    )


# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 简单地打印数据集信息和加载器的长度
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"训练集加载器的长度: {len(train_loader)}")
print(f"验证集加载器的长度: {len(val_loader)}")
print(f"测试集加载器的长度: {len(test_loader)}")

# 定义优化器 (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)  # 调整学习率和权重衰减

# 定义学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


def evaluate_model(model, data_loader, criterion, num_classes):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    return avg_loss, accuracy, cm, all_labels, all_preds


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', filename=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('实际标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def save_best_model(model, epoch, val_accuracy, output_dir):
    model_path = os.path.join(output_dir, f"best_model_epoch_{epoch+1}_acc_{val_accuracy:.4f}.pth")
    torch.save(model.state_dict(), model_path)  # 保存模型参数（推荐）
    # 或保存整个模型：torch.save(model, model_path)
    print(f"最佳模型已保存至：{model_path}")

def save_results_to_excel(all_labels, all_preds, dataset_name, loss, accuracy, cm, classes):
    data = {
        "实际标签": all_labels,
        "预测标签": all_preds,
    }
    df = pd.DataFrame(data)
    df["数据集"] = dataset_name
    df["损失"] = loss
    df["准确率"] = accuracy
    df["混淆矩阵"] = [str(cm)] * len(df)  # 存储混淆矩阵的字符串表示
    df["类别"] = [str(classes)] * len(df)  # 存储类别

    excel_file = os.path.join(output_dir, "model_results.xlsx")

    if not os.path.exists(excel_file):
        df.to_excel(excel_file, index=False)
    else:
        # 如果文件存在，则追加新数据
        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            start_row = writer.sheets["Sheet1"].max_row  # 获取当前表格的最大行数
            df.to_excel(writer, sheet_name="Sheet1", index=False, startrow=start_row)


def plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs, filename=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='训练准确率')
    plt.plot(epochs, val_accs, 'r-', label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('轮数')
    plt.ylabel('准确率')
    plt.legend()

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


# 训练循环
num_epochs = 30
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')  # 创建 tqdm 进度条
    for inputs, labels in progress_bar:
        optimizer.zero_grad()  # 清除之前的梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        total_train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_correct / total_samples
    train_losses.append(avg_train_loss)
    train_accs.append(train_accuracy)

    val_loss, val_accuracy, _, _, _ = evaluate_model(model, val_loader, criterion, num_classes)
    val_losses.append(val_loss)
    val_accs.append(val_accuracy)
    scheduler.step(val_loss) # 更新学习率
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 保存当前轮次的最佳模型
    if val_accuracy == max(val_accs):  # 或使用全局变量记录最佳准确率
        save_best_model(model, epoch, val_accuracy, output_dir)

# 绘制训练和验证的损失和准确率曲线
plot_loss_and_accuracy(train_losses, val_losses, train_accs, val_accs,
                       filename=os.path.join(images_dir, "training_loss_accuracy.png"))

# 评估验证集
val_loss, val_accuracy, val_cm, val_labels, val_preds = evaluate_model(model, val_loader, criterion, num_classes)
print(f"验证集损失: {val_loss:.4f}")
print(f"验证集准确率: {val_accuracy:.4f}")
plot_confusion_matrix(val_cm, train_dataset.classes, title='验证集混淆矩阵',
                      filename=os.path.join(images_dir, "val_confusion_matrix.png"))
save_results_to_excel(val_labels, val_preds, "验证集", val_loss, val_accuracy, val_cm, train_dataset.classes)

# 评估测试集
test_loss, test_accuracy, test_cm, test_labels, test_preds = evaluate_model(model, test_loader, criterion, num_classes)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")
plot_confusion_matrix(test_cm, train_dataset.classes, title='测试集混淆矩阵',
                      filename=os.path.join(images_dir, "test_confusion_matrix.png"))
save_results_to_excel(test_labels, test_preds, "测试集", test_loss, test_accuracy, test_cm, train_dataset.classes)

print("评估完成，结果和图像已保存")

# 导出前再次定义 device（和脚本开头保持一致）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 确保 model 也在这个 device
model = model.to(device)

# 然后再构造 dummy 并导出
dummy = torch.randn(1, 3, 224, 224, device=device)

# 3A. 直接导出 ONNX（opset16，batch 可变，空间维固定）
torch.onnx.export(
    model,                                    # 训练后的 nn.Module
    dummy,                                    # 示例输入
    "run/data3/output/resnet_model.onnx",                 # 存储路径
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"},  # 仅 batch 动态
                  "output": {0: "batch_size"}}
)
print("✅ ONNX 导出完成（直接导出）：output/resnet_model.onnx")
