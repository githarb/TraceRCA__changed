import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 生成50个1到10之间的均匀随机数
random_data = np.random.uniform(1, 1000, 5000)

# 创建一个图形
plt.figure(figsize=(8, 6))

# 使用Seaborn的distplot绘制概率密度函数图形
sns.distplot(random_data, hist=True, kde=True, bins=10, color='blue', label='Random Data')

# 设置 X 轴和 Y 轴的标签
plt.xlabel('Value')
plt.ylabel('PDF (Probability Density Function)')

# 添加图例
plt.legend()

# 设置图形标题
plt.title('Probability Density Function of Random Data')

# 显示图形
plt.show()
