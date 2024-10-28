import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# DataPreparation
# **********************************************************************************************************************
# labels = ***
# predictions = ***
labels = [1, 0, 1, 0, 1, 0, 1]
predictions = [0.1, 0.2, 0.3, 0.2, 0.8, 0.1, 0.3]
# **********************************************************************************************************************

# 计算 ROC 曲线的参数
fpr, tpr, thresholds = roc_curve(labels, predictions)
roc_auc = auc(fpr, tpr)

# 设置绘图参数
plt.figure(figsize=(8, 8))  # 正方形画布
plt.rcParams.update({'font.size': 14})  # 设置全局字体大小

# 绘制 ROC 曲线，增加线宽为3
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right", prop={'size': 14})
plt.grid(True)

# 显示图形
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()