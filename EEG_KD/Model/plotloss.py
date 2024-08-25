import matplotlib.pyplot as plt
import numpy as np

# 随机生成100个epoch的数据
epochs = range(1, 101)
train_losses_s = np.random.rand(100) * 0.5  # 随机生成训练损失_s，范围在0到0.5之间
train_losses_t = np.random.rand(100) * 0.5  # 随机生成训练损失_t，范围在0到0.5之间
val_losses_s = np.random.rand(100) * 0.4  # 随机生成验证损失_s，范围在0到0.4之间
val_losses_t = np.random.rand(100) * 0.4  # 随机生成验证损失_t，范围在0到0.4之间

# 创建一个figure对象，并设置大小
plt.figure(figsize=(10, 6))

# 绘制训练损失_s
plt.plot(epochs, train_losses_s, label='Train Loss (s)', linestyle='-', color='blue')

# 绘制训练损失_t
plt.plot(epochs, train_losses_t, label='Train Loss (t)', linestyle='--', color='green')

# 绘制验证损失_s
plt.plot(epochs, val_losses_s, label='Validation Loss (s)', linestyle='-', color='red')

# 绘制验证损失_t
plt.plot(epochs, val_losses_t, label='Validation Loss (t)', linestyle='--', color='orange')

# 添加标题和标签
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()  # 添加图例

# 显示图表
plt.grid(True)  # 添加网格线
plt.tight_layout()  # 自动调整布局
plt.show()
