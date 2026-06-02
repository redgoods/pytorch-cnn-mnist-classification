
# PyTorch 手写数字识别（MNIST CNN）

一个使用 **PyTorch** 实现的卷积神经网络（CNN）项目，在经典 MNIST 数据集上进行手写数字 0~9 的分类。模型结构简洁，代码易于理解和修改，适合 PyTorch 入门学习。

## 功能特性

- 使用 PyTorch 构建两层卷积 + 全连接层的 CNN 模型
- 完整的训练、测试、可视化流程
- 自动选择 GPU（若可用）或 CPU 进行训练
- 训练过程中实时显示损失和准确率
- 绘制训练曲线，直观判断模型收敛情况
- 保存和加载训练好的模型参数
- **测试集准确率 > 98%**

## 环境依赖

- Python 3.8+
- PyTorch >= 1.10（推荐 2.x）
- torchvision
- matplotlib

可以使用以下命令安装依赖（CPU 版 PyTorch，如需 GPU 版请访问 [pytorch.org](https://pytorch.org) 选择对应的 CUDA 版本）:

```bash
pip install torch torchvision matplotlib
```

## 文件结构

```
.
├── data/                 # 自动下载的 MNIST 数据集存放目录
|—— model/             
    ├── mnist_cnn.pth         # 训练后保存的模型参数文件（运行后生成）
├── main.py               # 主程序脚本（可将代码保存为此文件）
└── README.md             # 项目说明文档
```

## 快速开始

### 1. 克隆或下载本仓库

```bash
git clone <your-repo-url>   # 或直接下载代码
cd mnist-cnn
```

### 2. 安装依赖

```bash
pip install -r requirements.txt   # 如果没有 requirements.txt，手动执行上面的 pip 命令
```

### 3. 运行训练与测试

```bash
python main.py
```

程序会自动下载 MNIST 数据集（如果本地没有），然后开始训练。训练过程中会打印每个 epoch 的损失和准确率，并最终输出测试集准确率。

### 4. 查看结果

训练结束后会：
- 在控制台打印测试集准确率（例如：`测试集准确率: 99.18%`）
- 弹出窗口显示训练损失曲线和准确率曲线
- 弹出窗口显示部分测试样本的预测结果
- 在项目根目录生成 `mnist_cnn.pth` 模型文件

## 模型架构

```
Input (1x28x28)
  → Conv2d(1→16, 3x3, pad=1) → ReLU → MaxPool2d(2,2)
  → Conv2d(16→32, 3x3, pad=1) → ReLU → MaxPool2d(2,2)
  → Flatten
  → Linear(32*7*7 → 128) → ReLU
  → Linear(128 → 10)
```

- 两个卷积层逐渐提取特征，池化层降维
- 全连接层将特征映射到 10 个类别
- 使用交叉熵损失，无需在模型最后加 Softmax

## 训练参数

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 0.001 |
| 损失函数 | CrossEntropyLoss |
| 批大小 (batch_size) | 64 |
| 训练轮数 (epochs) | 10 |
| 数据预处理 | ToTensor + Normalize(mean=0.1307, std=0.3081) |

## 验收标准

- [x] 模型成功训练，无报错
- [x] 测试集准确率达到 **98%** 以上
- [x] 训练曲线正常：损失稳步下降，准确率上升后平稳，未出现过拟合/欠拟合
- [x] 模型能够正确预测示例手写数字图片
- [x] 能够保存并重新加载模型继续使用

## 示例输出

训练过程中日志示例：

```
Epoch [1/10], Step [100/938], Loss: 0.3469
...
Epoch [1 / 10], Train Loss: 0.0026, Train Acc: 0.9477
...
Epoch [10 / 10], Train Loss: 0.0001, Train Acc: 0.9973
测试集准确率: 99.18%
```

训练曲线示例（自动弹出窗口）：
- 左图：损失从约 0.2 降至接近 0
- 右图：准确率从 90%+ 上升至 99%+ 并趋于稳定

预测可视化：会显示 15 张测试图片，标题为 “真实: X, 预测: Y”，绝大多数应一致。

## 常见问题

**1. 下载 MNIST 数据集速度慢或失败？**  
可以手动下载数据集放入 `data/MNIST/raw/` 目录，具体文件参考 PyTorch 官方文档。

**2. 测试准确率未达到 98%？**  
- 检查是否正确使用 GPU（打印“使用设备: cuda”）
- 适当增加 epoch 数量（例如 15）
- 确认未修改代码中的随机种子或数据标准化参数

**3. 代码运行环境没有图形界面？**  
可以注释掉所有 `plt.show()` 行，或使用非交互式后端（如 `matplotlib.use('Agg')`）并保存图片。

## 改进方向（可选）

- 添加 Dropout 层防止过拟合
- 使用 Batch Normalization 加速收敛
- 引入学习率衰减策略
- 数据增强（随机旋转、平移）
- 用 TensorBoard 可视化训练过程
- 导出为 ONNX 格式并在其他平台推理
