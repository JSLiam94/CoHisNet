import matplotlib.pyplot as plt
import glob

# 定义存储所有数据的字典
data_dict = {}

# 获取所有 txt 文件的路径
file_paths = glob.glob('val*.txt')  # 读取当前目录下所有 txt 文件

# 遍历每个文件并提取数据
for file_path in file_paths:
    epochs = []
    val_acc = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('epoch :'):
                # 提取 epoch 值
                epoch = int(line.split(':')[1].strip())
                epochs.append(epoch)
            elif line.startswith('val_acc:'):
                # 提取 val_acc 值
                acc = float(line.split(':')[1].strip())
                val_acc.append(acc)
    
    # 检查 epochs 和 val_acc 的长度是否一致
    if len(epochs) != len(val_acc):
        print(f"Warning: {file_path} has inconsistent data (epochs: {len(epochs)}, val_acc: {len(val_acc)}). Filling with NaN.")
        # 填充较短的列表为 NaN
        max_length = max(len(epochs), len(val_acc))
        epochs.extend([None] * (max_length - len(epochs)))  # 填充 epochs
        val_acc.extend([None] * (max_length - len(val_acc)))  # 填充 val_acc
    
    # 将数据存储到字典中，键为文件名
    data_dict[file_path] = {'epochs': epochs, 'val_acc': val_acc}

# 绘制多条曲线
plt.figure(figsize=(10, 6))  # 设置图表大小

for file_path, data in data_dict.items():
    line, = plt.plot(data['epochs'], data['val_acc'], marker='o', linestyle='-', label=file_path)
    
    # 找到每个 Epoch 的最大值
    max_val_acc = max(data['val_acc']) if data['val_acc'] else None
    if max_val_acc is not None:
        # 找到最大值对应的 Epoch
        max_epoch = data['epochs'][data['val_acc'].index(max_val_acc)]
        # 在最高点附近显示数值
        plt.text(max_epoch, max_val_acc, f'{max_val_acc:.5f}', fontsize=7, color=line.get_color(),
                 ha='center', va='bottom')  # 在数据点上方显示数值

# 添加标题和标签
plt.title('Epoch vs Validation Accuracy (Multiple Files)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (val_acc)')

# 显示网格
plt.grid(True)

# 显示图例
plt.legend()

# 显示图表
plt.show()
