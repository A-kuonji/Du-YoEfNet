# Du-YoEfNet (Dual-Attention YOLO–EfficientNet Fusion Network)

A Dual-Attention YOLO–EfficientNet Feature Fusion Network for Artistic Painting Recognition

# Abstract:

The digitization of artistic paintings has emerged as a major advancement, enhancing efficiency and accuracy in collection management and painting authentication. However, automated painting recognition based on computer vision remains challenging due to subtle stylistic differences and complex visual patterns across various artists. Therefore, this study proposed a novel dual-attention feature fusion network, named Du-YoEfNet based on deep learning, which used a network fusion mechanism to extract complementary features. The model integrated two modules, Squeeze-and-Excitation Network and Convolutional Block Attention Module, corresponding to channel attention and spatial attention mechanisms, respectively. This approach enabled the effective capture of both global compositional information and local brushstroke details for more discriminative feature representations. Three datasets were constructed for extensive experimentation. The results demonstrated that the proposed model achieved excellent performance across all datasets and outperformed seven current mainstream classification networks. Notably, Du-YoEfNet maintained exceptional recognition capability even.

# Keywords: 

artistic paintings; feature fusion; deep learning; artificial intelligence; art historical

# Dataset

This study uses a self-collected dataset.

Due to its large size, the dataset is not included in this repository.

# Environment

- OS: Windows 11
- Python: 3.8
- PyTorch: 2.5.1
- CUDA: 12.4
- GPU: NVIDIA RTX 3060 (6GB)

# Installation

pip install -r requirements.txt

# Run

python EfficientNet YOLOv11 Feature-level fusion.py

# License

This code is released under the BSD 3-Clause License.

# Notes for Reviewers

The repository is provided for reproducibility purposes.
All hyperparameters used in the paper are defined in the config files.
