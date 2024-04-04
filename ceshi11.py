import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB



def GuaussianNB():
    clf = GaussianNB()
    return clf

def Svm():
    clf = svm.SVC()
    return clf

def KNN():
    clf = neighbors.KNeighborsClassifier()
    return clf



def getData_3(x):
    fPath = 'D:\E盘\数据\microarray data\Lung-200.csv'
    dataMatrix = np.array(
        pd.read_csv(fPath, header=None, skiprows=1))
    rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
    sampleData = []
    sampleClass = []
    double = []
    temp = []

    a = np.array(x)

    for i in range(0, rowNum):
        for j in a:
            temp.append(dataMatrix[i, j])
        temp.append(dataMatrix[i, -1])
        double.append(temp)
        temp = []
    double = np.array(double)

    for i in range(0, rowNum):
        tempList = list(double[i, :])
        sampleClass.append(tempList[-1])
        sampleData.append(tempList[:-1])
    sampleM = np.array(sampleData)  # 二维矩阵，一行是一个样本，行数=样本总数，列数=样本特征数
    classM = np.array(sampleClass)  # 一维列向量，每个元素对应每个样本所属类别
    skf = StratifiedKFold(n_splits=10)
    setDict = {}  # 创建字典，用于存储生成的训练集和测试集
    count = 1

    for trainI, testI in skf.split(sampleM, classM):
        trainSTemp = []  # 用于存储当前循环抽取出的训练样本数据
        trainCTemp = []  # 用于存储当前循环抽取出的训练样本类标
        testSTemp = []  # 用于存储当前循环抽取出的测试样本数据
        testCTemp = []  # 用于存储当前循环抽取出的测试样本类标
        # 生成训练集
        trainIndex = list(trainI)
        for t1 in range(0, len(trainIndex)):
            trainNum = trainIndex[t1]
            trainSTemp.append(list(sampleM[trainNum, :]))
            trainCTemp.append(list(classM)[trainNum])
        setDict[str(count) + 'train'] = np.array(trainSTemp)
        setDict[str(count) + 'trainclass'] = np.array(trainCTemp)
        # 生成测试集
        testIndex = list(testI)
        for t2 in range(0, len(testIndex)):
            testNum = testIndex[t2]
            testSTemp.append(list(sampleM[testNum, :]))
            testCTemp.append(list(classM)[testNum])
        setDict[str(count) + 'test'] = np.array(testSTemp)
        setDict[str(count) + 'testclass'] = np.array(testCTemp)
        count += 1
    return(setDict)


def getRecognitionRate(testPre, testClass):
    testNum = len(testPre)
    rightNum = 0
    for i in range(0, testNum):
        if testClass[i] == testPre[i]:
            rightNum += 1
    return float(rightNum) / float(testNum)


def cal1(x, setNums):
        clf_RF3 = GaussianNB()
        RF3_rate = 0.0
        #setDict = getData_3()
        #setNums = len(setDict.keys()) / 4
        for i in range(1, int(setNums + 1)):
            trainMatrix = x[str(i) + 'train']
            trainClass = x[str(i) + 'trainclass']
            testMatrix = x[str(i) + 'test']
            testClass = x[str(i) + 'testclass']
            clf_RF3.fit(trainMatrix, trainClass)
            RF3_rate += getRecognitionRate(clf_RF3.predict(testMatrix), testClass)
            C = RF3_rate / float(setNums)
        return(C)

def result(x):
    setDict = getData_3(x)
    setNums = len(setDict.keys()) / 4
    w = cal1(setDict, setNums)
    return(w)
