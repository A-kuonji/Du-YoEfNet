import warnings
# warnings.filterwarnings('ignore')
from ultralytics import YOLO


# 对于分类任务（task: classify），data参数应直接指向数据集路径，而不是 YAML 文件。
if __name__ == '__main__':
    # 不加载预训练权重
    model = YOLO(model=r'D:\project\EfficientNet_YOLOv11\ultralytics\cfg\models\11\yolo11-cls-se.yaml')  # 创建 YOLO 模型实例，使用指定的模型配置文件
    # print(model)  # 打印模型结构
    model.train(data=r'datasets',         # 指定训练数据集的路径
                imgsz=640,                # 指定输入图像的大小为 640x640 像素
                epochs=100,               # 指定训练的总轮数
                batch=40,                 # 指定每个批次的大小为 40 张图像
                workers=0,                # 指定用于数据加载的 worker 数量为 0 (主线程加载)
                device='cuda',            # 指定使用的设备，默认为 CUDA (如果可用)，否则使用 CPU
                optimizer='SGD',          # 指定使用的优化器为随机梯度下降 (SGD)
                close_mosaic=10,          # 在最后 10 个 epoch 关闭 Mosaic 数据增强
                resume=False,             # 如果为 True，则从上次中断的地方恢复训练。这里设置为 False，表示不恢复训练。
                project='runs/train',     # 指定训练结果保存的父目录
                name='test',              # 指定本次训练实验的名称
                single_cls=False,         # 如果为 True，则将所有类别视为单个类别。这里设置为 False，表示多类别训练。
                cache=False,              # 如果为 True，则将图像加载到内存中进行缓存，加速训练。这里设置为 False，表示不使用缓存。
                pretrained=False,         # 如果为 True，则加载预训练权重。这里设置为 False，表示不使用预训练权重，从头开始训练。
                amp=False,                # 添加这一行，禁用 AMP (Automatic Mixed Precision) 自动混合精度训练。启用 AMP 可以加速训练并减少显存占用，但某些情况下可能会导致精度下降。
                )
