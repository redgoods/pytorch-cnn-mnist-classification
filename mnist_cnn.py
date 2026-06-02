# ============================================
# 项目：PyTorch 实现 CNN 手写数字分类（MNIST）
# 包含数据加载、模型构建、训练、测试、可视化、模型保存与加载
# ============================================


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 环境准备与设备选择
# 检测是否有可用的 GPU，有则使用 cuda，否则使用 cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2.数据加载与预处理
#数据转换器, 定义数据转换流程：转为张量 + 标准化（MNIST 的全局均值和标准差）
transform = transforms.Compose([
    transforms.ToTensor(),                                       # 将 PIL 图像转为 Tensor，并将像素值缩放到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))        # 标准化：减去均值 0.1307，除以标准差 0.3081
])

# 下载并加载 MNIST 训练集
train_dataset = datasets.MNIST(
    root='./data',               # 数据存放目录
    train=True,                   # True 表示训练集
    download=True,                  # 如果没有数据则下载
    transform=transform,           # 应用上面定义的预处理数据转换器
)

# 下载并加载 MNIST 测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,            # False 表示测试集
    download=True,
    transform=transform
)

# 创建数据加载器，用于批量读取数据
# batch_size=64：每次训练使用 64 张图片
# shuffle=True：训练时打乱数据顺序，避免模型记住顺序
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) # 测试集不需要打乱顺序
print(f"训练集大小：{len(train_dataset)}")
print(f"测试集大小：{len(test_dataset)}")

# 可视化一个 batch 中的第一张图片，确认数据加载正确
example_data, example_targets = next(iter(train_loader))
plt.imshow(example_data[0][0], cmap='gray')         # 显示灰度图
plt.title(f"标签：{example_targets[0].item()}")        # 显示对应的数据标签
plt.show()

# 3. 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # ----- 第一个卷积块 -----
        # 输入通道数 1（灰度图），输出通道数 16，卷积核大小 3x3，padding=1 保持尺寸不变
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()                      # ReLU 激活函数
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 最大池化，尺寸减半

        # ----- 第二个卷积块 -----
        # 输入通道数 16（来自上一层），输出通道数 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- 全连接层 -----
        # 经过两次池化后，图像尺寸从 28x28 变为 7x7，通道数为 32
        # 展平后特征数为 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)       # 第一个全连接层，输出 128 个特征
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)                # 输出层，10 个类别对应数字 0-9

    def forward(self, x):
        # 输入 x 的形状: (batch_size, 1, 28, 28)
        out = self.conv1(x)      # → (batch_size, 16, 28, 28)
        out = self.relu1(out)
        out = self.pool1(out)    # → (batch_size, 16, 14, 14)

        out = self.conv2(out)    # → (batch_size, 32, 14, 14)
        out = self.relu2(out)
        out = self.pool2(out)    # → (batch_size, 32, 7, 7)

        # 将多维张量展平为二维，第一维是 batch_size，第二维是特征
        out = out.view(out.size(0), -1)   # → (batch_size, 32*7*7)
        out = self.fc1(out)               # → (batch_size, 128)
        out = self.relu3(out)
        out = self.fc2(out)               # → (batch_size, 10)
        return out

# 实例化模型，并将模型移动到计算设备（GPU/CPU）
model = CNN().to(device)
print(model)

# 4. 定义损失函数和优化器
# 交叉熵损失，内部会自动对模型输出计算 softmax，因此模型最后一层不需要 softmax
criterion = nn.CrossEntropyLoss()
# Adam 优化器, lr为学习率
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练模型
epochs = 10                     # 训练轮数
total_step = len(train_loader)      # 每个 epoch 的批次数量
train_losses = []                   # 记录每个 epoch 的平均损失
train_accs = []                     # 记录每个 epoch 的训练准确率

for epoch in range(epochs):
    model.train()
    running_loss = 0.0  # 累计损失
    correct = 0  # 累计正确预测个数
    total = 0  # 累计样本总数

    for i, (images, labels) in enumerate(train_loader):
        # 将数据和标签移动到计算设备
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()          # 梯度清零(上一次梯度)
        loss.backward()                # 反向传播
        optimizer.step()               # 梯度更新

        # 统计信息
        running_loss += loss.item()  # 累加损失值（loss.item() 为 Python 数字）
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果中概率最大的类别索引
        total += labels.size(0)  # 累加当前 batch 的样本数
        correct += (predicted == labels).sum().item()  # 累加预测正确的个数

        # 每 100 个 batch 打印一次当前步的损失
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1} / {epochs}, Step [{i + 1} / {total_step}], Loss: {loss.item():.4f}')

    # 计算并记录该 epoch 的平均损失和准确率
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f'Epoch [{epoch + 1} / {epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

# 绘制训练过程中的损失曲线和准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.title('训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='训练准确率')
plt.title('训练准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 6. 测试模型性能
model.eval()
correct = 0
total = 0

# 测试时不计算梯度，节省显存和计算
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'测试集准确率: {100 * correct / total:.2f}%')

# 可视化部分测试样本的预测结果
# 取出一个 batch 的测试数据
example_data, example_targets = next(iter(test_loader))
example_data = example_data.to(device)
example_targets = example_targets.to(device)
outputs = model(example_data)
_, predicted = torch.max(outputs, 1)

# 绘制前 15 张图片，标题显示真实标签和预测标签
plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(example_data[i][0].cpu(), cmap='gray')   # 将数据移回 CPU 用于 matplotlib 显示
    plt.title(f'真实: {example_targets[i].item()}, 预测: {predicted[i].item()}')
    plt.axis('off')    # 不显示坐标轴
plt.show()

# 7. 保存与加载模型
# 保存模型的参数（state_dict），不保存整个模型结构
torch.save(model.state_dict(), './model/mnist_cnn.pth')
print("模型已保存为 ./model/mnist_cnn.pth")


# 演示加载模型：先重新创建一个相同结构的模型，然后加载参数
model = CNN().to(device)
model.load_state_dict(torch.load('model/mnist_cnn.pth'))
model.eval()   # 加载后设为评估模式
print("模型加载成功，可以继续使用。")