import os
import math
import argparse
import csv  # 用于CSV记录

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate

# 设置matplotlib非交互后端，避免 GUI 错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_and_log_confusion_matrix(y_true, y_pred, class_names, epoch, tb_writer, save_dir):
    """
    绘制混淆矩阵并保存，同时记录到 TensorBoard
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted',
        ylabel='True',
        title=f'Confusion Matrix Epoch {epoch}'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()

    # 保存本地
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f'cm_epoch_{epoch}.png')
    fig.savefig(fig_path)
    plt.close(fig)

    # TensorBoard记录
    tb_writer.add_figure('confusion_matrix', fig, epoch)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建目录
    os.makedirs("./weights", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # TensorBoard
    tb_writer = SummaryWriter(log_dir="./logs/tensorboard")

    # CSV日志
    csv_path = os.path.join("./logs", "training_log.csv")
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"])

    # 读取数据
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    class_names = [str(i) for i in range(args.num_classes)]

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights:
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights = torch.load(args.weights, map_location=device)
        weights_dict = weights.get('model', weights)
        for k in list(weights_dict.keys()):
            if 'head' in k:
                weights_dict.pop(k)
        model.load_state_dict(weights_dict, strict=False)
        print("Loaded pretrained weights, excluding head.")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print(f"Training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5e-2)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # 收集预测用于混淆矩阵
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        model.train()

        # 绘制并记录混淆矩阵
        plot_and_log_confusion_matrix(all_labels, all_preds, class_names,
                                      epoch, tb_writer, save_dir="./logs/confusion_matrix")

        lr = optimizer.param_groups[0]["lr"] if hasattr(optimizer, 'param_groups') else None
        # TensorBoard 日志
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        if lr is not None:
            tb_writer.add_scalar("learning_rate", lr, epoch)

        # CSV 日志
        csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])
        csv_file.flush()

        # 保存模型
        torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")

    csv_file.close()
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=28)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str, default="./data/data3")
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. cuda:0 or cpu)')
    opt = parser.parse_args()
    main(opt)
