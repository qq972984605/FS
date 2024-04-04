import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, recall_score, f1_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def getData_3(x):
    fPath = 'D:/E盘/数据/microarray data/Colon-F200.csv'
    dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))
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
    sampleM = np.array(sampleData)
    classM = np.array(sampleClass)
    skf = StratifiedKFold(n_splits=10)
    setDict = {}

    count = 1
    for trainI, testI in skf.split(sampleM, classM):
        trainSTemp = []
        trainCTemp = []
        testSTemp = []
        testCTemp = []

        trainIndex = list(trainI)
        for t1 in range(0, len(trainIndex)):
            trainNum = trainIndex[t1]
            trainSTemp.append(list(sampleM[trainNum, :]))
            trainCTemp.append(list(classM)[trainNum])
        setDict[str(count) + 'train'] = np.array(trainSTemp)
        setDict[str(count) + 'trainclass'] = np.array(trainCTemp)

        testIndex = list(testI)
        for t2 in range(0, len(testIndex)):
            testNum = testIndex[t2]
            testSTemp.append(list(sampleM[testNum, :]))
            testCTemp.append(list(classM)[testNum])
        setDict[str(count) + 'test'] = np.array(testSTemp)
        setDict[str(count) + 'testclass'] = np.array(testCTemp)
        count += 1
    return setDict


def getRecognitionRate(testPre, testClass):
    testNum = len(testPre)
    rightNum = 0
    for i in range(0, testNum):
        if testClass[i] == testPre[i]:
            rightNum += 1
    return float(rightNum) / float(testNum)


def cal(x, setNums):
    clf_RF = GaussianNB()
    RF_rate = 0.0
    AUC = 0.0
    recall = 0.0
    f = 0.0
    precision = 0.0

    for i in range(1, int(setNums + 1)):
        X_train = x[str(i) + 'train']
        y_train = x[str(i) + 'trainclass']
        X_test = x[str(i) + 'test']
        y_test = x[str(i) + 'testclass']
        clf_RF.fit(X_train, y_train)
        RF_rate += getRecognitionRate(clf_RF.predict(X_test), y_test)

        y_scores = clf_RF.predict_proba(X_test)[:, 1]
        AUC += roc_auc_score(y_test, y_scores)
        recall += recall_score(clf_RF.predict(X_test), y_test)
        f += f1_score(clf_RF.predict(X_test), y_test)
        precision += precision_score(clf_RF.predict(X_test), y_test)

    A = RF_rate / float(setNums)
    B = AUC / float(setNums)
    C = recall / float(setNums)
    D = f / float(setNums)
    E = precision / float(setNums)
    return A, B, C, D, E


def plot_roc_curve(mean_fpr, mean_tpr, mean_auc):
    plt.figure()
    lw = 2
    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % mean_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Colon')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    x = [74, 80, 102, 108, 128, 185, 192, 199]
    setDict = getData_3(x)
    setNums = len(setDict.keys()) / 4
    A, B, C, D, E = cal(setDict, setNums)
    print(f"Recognition Rate: {A}")
    print(f"AUC: {B}")
    print(f"Recall: {C}")
    print(f"F1 Score: {D}")
    print(f"Precision: {E}")

    # Calculate mean FPR and TPR
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for i in range(1, int(setNums + 1)):
        X_train = setDict[str(i) + 'train']
        y_train = setDict[str(i) + 'trainclass']
        X_test = setDict[str(i) + 'test']
        y_test = setDict[str(i) + 'testclass']

        clf_RF = GaussianNB()
        clf_RF.fit(X_train, y_train)
        y_score = clf_RF.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    # Compute mean ROC curve and AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # Plot ROC curve
    plot_roc_curve(mean_fpr, mean_tpr, mean_auc)
