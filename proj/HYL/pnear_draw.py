import os
import matplotlib.pyplot as plt
import numpy as np
from wcode.cycpep.fold import extract_pnear


names = os.listdir('/mnt/c/tmp/PRCyc')
for name in names:

    # 示例数据
    timestamps = np.array([100, 300, 1000, 3000])
    data = []
    for i, t in enumerate(timestamps):
        data.append([])
        for j in range(10):
            path = os.path.join(f'/mnt/c/tmp/PRCyc/{name}/{t}/{j}/')
            p_n = extract_pnear(path)
            data[i].append(p_n)

    # 计算均值和标准差
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]

    # 创建图表
    plt.figure(figsize=(8, 6))

    # 生成平滑的背景颜色带
    plt.fill_between(timestamps,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     color='lightcoral', alpha=0.3)

    # 绘制数据点和均值线
    for i in range(len(timestamps)):
        plt.scatter([timestamps[i]] * len(data[i]), data[i], color='blue')

    plt.plot(timestamps, means, color='black', marker='o')
    plt.ylim(0, 1)

    # 设置坐标轴标签
    plt.xlabel('Timestamp')
    plt.ylabel('Value')

    # 显示图表
    plt.title('Data Distribution Over Time')
    plt.savefig(f'/mnt/c/tmp/{name}.png')
