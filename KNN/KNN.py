import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
fruits_df = pd.read_csv('fruit_data_with_colors.txt', sep='\t')

# Data Preview
fruits_df.head()

print('The number of sample：', len(fruits_df))

# 分类直方图
sns.countplot(fruits_df['fruit_name'], label="Count")

# 创建目标标签和名称的字典
fruit_name_dict = dict(zip(fruits_df['fruit_label'], fruits_df['fruit_name']))

# 划分数据集
X = fruits_df[['mass', 'width', 'height', 'color_score']]
y = fruits_df['fruit_label']

# 划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

print('数据集样本数：{}，训练集样本数：{}，测试集样本数：{}'.format(len(X), len(X_train), len(X_test)))

# 查看数据集
sns.pairplot(data=fruits_df, hue='fruit_name', vars=['mass', 'width', 'height', 'color_score'])


# 建立KNN模型
knn = KNeighborsClassifier(n_neighbors=5)

# 拟合模型
knn.fit(X_train, y_train)

# 测试模型
y_pred = knn.predict(X_test)
print('预测标签：', y_pred)

print('真实标签: ', y_test.values)

# 计算该模型的准确率
acc = accuracy_score(y_test, y_pred)
print('准确率: ', acc)


# 查看不同K值的情况下，对模型准确度的影响
k_range = range(1, 20)
acc_score_list = []

for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc_score_list.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(k_range, acc_score_list, marker='o')
plt.xticks([0, 5, 10, 15, 20])
plt.show()
