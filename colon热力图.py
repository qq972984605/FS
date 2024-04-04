import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import palettable

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文



fPath = 'D:\E盘\数据\论文图\Colon迭代\Colon-.csv'
dataMatrix = np.array(
    pd.read_csv(fPath, header=None, skiprows=0))
rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
sampleData = []
sampleClass = []
for i in range(0, rowNum):
    tempList = list(dataMatrix[i, :])
    sampleClass.append(tempList[-1])
    sampleData.append(tempList[:-1])
sampleM = np.array(sampleData)  # 二维矩阵，一行是一个样本，行数=样本总数，列数=样本特征数
classM = np.array(sampleClass)  # 一维列向量，每个元素对应每个样本所属类别


pd_L = pd.DataFrame(np.hstack((sampleM, classM.reshape(62, 1))),columns=['X87159', 'M76378', 'R36977', 'L08069', 'T47377', 'H55916', 'T57468','class'])
pd_L1 = pd_L.drop(['class'],axis=1)
row_c = dict(zip(pd_L['class'].unique(), ['lime','yellow']))
sns.clustermap(data=pd_L1, col_cluster=False, row_colors=pd_L['class'].map(row_c), cmap='Reds',figsize=(8, 10),standard_scale=1, method='average')

plt.show()



'lime','yellow','black','deepskyblue','coral'