import sys
import io
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from tqdm import tqdm
import torch.nn.functional as F

# --------------------------
# 配置参数（务必与训练代码一致）
# --------------------------
FUSION_MODEL_PATH = 'runs/output/data3/fusion_best.pth'  # 融合模型权重路径
VAL_DIR = 'datasets/data3/val'
CSV_OUTPUT = 'fusion_model_evaluation.csv'
MODEL_NAME = 'yolov11_efficientnet_fusion'
INPUT_SIZE = 400  # 与训练时输入尺寸一致
NUM_CLASSES = 28
NUM_RUNS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 修复中文输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --------------------------
# 数据预处理（与训练一致）
# --------------------------
transform_val = transforms.Compose([
    transforms.Resize(INPUT_SIZE + 32),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform_val)
val_loader = DataLoader(
    val_dataset,
    batch_size=40,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
CLASS_NAMES = val_dataset.classes
num_classes = len(CLASS_NAMES)


# --------------------------
# 模型定义（与训练代码完全一致）
# --------------------------
class YOLOv11FeatureExtractor(nn.Module):
    def __init__(self, yolo_weights='runs/train/data3_yolo11_se_cbam/weights/best.pt', use_neck=False):
        super().__init__()
        from ultralytics import YOLO  # 确保导入位置正确
        yolo_model = YOLO(yolo_weights).model
        self.backbone = nn.Sequential(*list(yolo_model.model))
        self.use_neck = use_neck
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = x
        feats = []
        for idx, layer in enumerate(self.backbone):
            out = layer(out)
            if (not self.use_neck and idx in {6, 14, 20}) or (self.use_neck and idx in {24, 28, 32}):
                feats.append(out)
        return feats[0]


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        if weights_path:
            net = torch.load(weights_path, map_location='cpu')
        else:
            raise ValueError("EfficientNet 权重路径不能为空")
        self.features = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        return self.features(x)


class FeatureAlign(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.target = size

    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, size=self.target, mode='bilinear', align_corners=False)


class FusionClassifier(nn.Module):
    def __init__(self, yolov11_cfg: dict, eff_cfg: dict, num_classes: int):
        super().__init__()
        self.yolo_feat = YOLOv11FeatureExtractor(**yolov11_cfg)
        self.eff_feat = EfficientNetFeatureExtractor(**eff_cfg)

        # 计算特征尺寸
        dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        y = self.yolo_feat(dummy)
        e = self.eff_feat(dummy)
        C_out = 256
        size = (y.shape[2], y.shape[3])

        self.align_y = FeatureAlign(y.shape[1], C_out, size)
        self.align_e = FeatureAlign(e.shape[1], C_out, size)
        self.head = nn.Sequential(
            nn.Conv2d(C_out * 2, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        y = self.yolo_feat(x)
        e = self.eff_feat(x)
        y_align = self.align_y(y)
        e_align = self.align_e(e)
        fused = torch.cat([y_align, e_align], dim=1)
        return self.head(fused)


# --------------------------
# 模型加载（关键修正）
# --------------------------
def load_fusion_model():
    # 1. 创建模型结构（必须先有结构才能加载权重）
    yolov11_cfg = {
        'yolo_weights': 'runs/train/data3_yolo11_se_cbam/weights/best.pt',
        'use_neck': False
    }
    eff_cfg = {
        'weights_path': 'runs/train/efficientnet/data3_efficientnet_100.pth'
    }

    model = FusionClassifier(
        yolov11_cfg=yolov11_cfg,
        eff_cfg=eff_cfg,
        num_classes=NUM_CLASSES
    )

    # 2. 检查权重文件是否存在
    if not os.path.exists(FUSION_MODEL_PATH):
        print(f"错误：模型权重文件不存在 -> {FUSION_MODEL_PATH}")
        sys.exit(1)

    # 3. 加载权重字典
    try:
        state_dict = torch.load(FUSION_MODEL_PATH, map_location=device)

        # 调试：打印权重类型（必须是OrderedDict）
        print(f"权重类型: {type(state_dict)}")

        # 4. 将权重加载到模型结构中
        model.load_state_dict(state_dict)
        print("权重加载成功")

    except Exception as e:
        print(f"权重加载失败: {str(e)}")
        print("可能原因：模型结构与权重不匹配")
        sys.exit(1)

    # 5. 移动模型到设备
    model = model.to(device)
    model.eval()  # 评估模式
    return model


# --------------------------
# 评估函数
# --------------------------
def evaluate(model, dataloader):
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return np.array(all_labels), np.array(all_preds), correct / total


# --------------------------
# 主函数
# --------------------------
if __name__ == '__main__':
    metrics_list = []

    for run in range(NUM_RUNS):
        print(f"\n===== 第 {run + 1}/{NUM_RUNS} 次评估 =====")
        model = load_fusion_model()  # 加载模型
        labels, preds, _ = evaluate(model, val_loader)

        # 计算指标
        cm = confusion_matrix(labels, preds, labels=range(num_classes))
        TP = np.diag(cm)
        FN = cm.sum(axis=1) - TP
        FP = cm.sum(axis=0) - TP
        TN = cm.sum() - (TP + FP + FN)
        eps = 1e-7

        class_acc = (TP / (TP + FN + eps)).mean() * 100
        inst_acc = (TP.sum() / cm.sum()) * 100
        sen = recall_score(labels, preds, average='macro') * 100
        pre = precision_score(labels, preds, average='macro') * 100
        f1 = f1_score(labels, preds, average='macro') * 100
        spe = (TN / (TN + FP + eps)).mean() * 100

        metrics_list.append({
            "Model": MODEL_NAME,
            "ClassAcc": class_acc,
            "InstAcc": inst_acc,
            "Sensitivity": sen,
            "Precision": pre,
            "Recall": sen,
            "Specificity": spe,
            "F1-score": f1
        })

    # 保存结果
    df = pd.DataFrame(metrics_list)
    grouped = df.groupby("Model").agg(['mean', 'std'])
    formatted = pd.DataFrame()
    for metric in grouped.columns.levels[0]:
        mean = grouped[metric]['mean']
        std = grouped[metric]['std']
        formatted[metric] = mean.map('{:.2f}'.format) + ' ± ' + std.map('{:.2f}'.format)

    formatted.index.name = "Models"
    formatted = formatted.reset_index()
    formatted = formatted[[
        "Models", "ClassAcc", "Sensitivity", "Precision",
        "Recall", "Specificity", "F1-score", "InstAcc"
    ]]
    formatted.columns = ["Models", "Acc", "Sen", "Pre", "Rec", "Spe", "F1-sc", "Overall Acc"]
    formatted.to_csv(CSV_OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果保存至 {CSV_OUTPUT}")
