import pandas as pd
import numpy as np
import random
import sys
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataSet = pd.read_csv("A2_t2_dataset.tsv", delimiter="\t", header=None)
dataSet.astype(float).values.tolist()

data = dataSet.iloc[:, :-1]
classification = dataSet.iloc[:, -1]

correlation = data.corr()
columns = np.full((correlation.shape[0],), True, dtype=bool)
for i in range(correlation.shape[0]):
    for j in range(i + 1, correlation.shape[0]):
        if correlation.iloc[i, j] >= 0.4:
            if columns[j]:
                columns[j] = False

selected_columns = data.columns[columns]
data = data[selected_columns]
datafinal = pd.DataFrame(data)

# print(len(selected_columns))
datafinal.insert(loc=len(selected_columns), column="class", value=classification)

finalData = datafinal.astype(float).values.tolist()
random.shuffle(finalData)
finalDF = pd.DataFrame(finalData)

X = finalDF.iloc[:, :-1]
X.astype(float).values.tolist()
Y = finalDF.iloc[:, -1]
Y.astype(float).values.tolist()

def perfMetrics(mat):
    n = len(mat)
    di = mat.diagonal(0)
    r = pd.Index(["C1", "C2"])
    c = pd.Index(["P", "R", "Sp", "FDR"])

    confMat = pd.DataFrame(mat, index=r, columns=r)
    RS = []
    for i in mat:
        sum = np.sum(i)
        RS.append(sum)
    fp = RS - di
    # false negative

    CS = []
    for i in range(n):
        sum = np.sum(mat[:, i])
        CS.append(sum)
    fn = CS - di

    tn = []
    sumofmat = mat.sum()
    for i in range(n):
        rowSum = np.sum(mat[i])
        calSum = np.sum(mat[:, i]) - di[i]
        truenegative = sumofmat - (rowSum + calSum)
        tn.append(truenegative)

    P = []
    for i in range(n):
        precision = di[i] / RS[i]
        P.append(precision)

    # Accuracy
    TNTP = np.sum(di)
    PN = np.sum(mat)
    Accuracy = TNTP / PN

    rcl = []
    for i in range(n):
        TPFN = di[i] + fn[i]
        recal = di[i] / TPFN
        rcl.append(recal)

    spcty = []
    for i in range(n):
        FPTN = fp[i] + tn[i]
        specificity = tn[i] / FPTN
        spcty.append(specificity)

    # FDR
    FDR = []
    for i in range(n):
        FPTP = fp[i] + di[i]
        falseDisR = fp[i] / FPTP
        FDR.append(falseDisR)

    perfMat = np.array([P, rcl, spcty, FDR]).transpose()
    perfDataFrame = pd.DataFrame(perfMat, index=r, columns=c)
    print(perfDataFrame)


overallAccuracy = []
overallFeatureSize = []

def cross_validation(k):
    init = 0
    folds = 5
    next = int(len(finalData) / folds)
    totNext = int(len(finalData) / folds)

    featuresArray = []
    accuracyArray = []
    print("---------------------------------------------------------------------")
    print("Number of neighbors: ", k)
    print("---------------------------------------------------------------------")
    for j in range(folds):

        X_train = pd.concat([X[:init], X[next:]])
        Y_train = pd.concat([Y[:init], Y[next:]])
        # print("init : ", init, "next : ", next)
        X_test = X[init:next]
        Y_test = Y[init:next]

        init = next
        next = next + totNext

        X_trainDF = pd.DataFrame(X_train, index=None)
        X_testDF = pd.DataFrame(X_test, index=None)
        colLen = X_trainDF.shape[1]

        selected_features = []
        index = 1
        accuracy = []
        accuracyVal = np.array(accuracy)
        for i in range(colLen):
            selected_features_train = X_trainDF.iloc[:, :index]
            selected_features_test = X_testDF.iloc[:, :index]
            index += 1

            train_set = np.array(selected_features_train)
            test_set = np.array(selected_features_test)

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(train_set, Y_train)

            prediction = knn.predict(test_set)
            accuracy.append(knn.score(test_set, Y_test))
        maxAccFeatures = np.argmax(accuracy) + 1
        maxAcc = np.amax(accuracy)
        featuresArray.append(maxAccFeatures)
        accuracyArray.append(maxAcc)
    print("features with max accuracy: ",featuresArray)
    print("Max accuracy in folds: ",accuracyArray)

    maxBestFeatind = np.argmax(accuracyArray)
    maxBestAcc = np.amax(accuracyArray)
    maxBestFeat = np.take(featuresArray,maxBestFeatind)
    overallAccuracy.append(maxBestAcc)
    overallFeatureSize.append(maxBestFeat)

    print("Selected feature: ", maxBestFeat)


testrange = 5
K_Value = 3

for test in range(testrange):
    cross_validation(K_Value)
    K_Value+=3

bestAcc = np.amax(overallAccuracy)
bestFeat = np.take(overallFeatureSize,np.argmax(overallAccuracy))
chosen_K = (np.argmax(overallAccuracy) +1) *3
print("overall feature sizes selected :", overallFeatureSize)
print("overall accuracy: ", overallAccuracy)
print("Best acc: ", bestAcc)
print("Best feature size: ", bestFeat)
print("Neighbors: ",chosen_K)
print("-------------------------Performance metrics-------------------------------")

def perf_Comp(neigh,feat):
    initX = 0
    foldsX = 5
    nextX = int(len(finalData) / foldsX)


    X_trainX = X[nextX:]
    Y_trainX = Y[nextX:]
    X_testX = X[initX:nextX]
    Y_testX = Y[initX:nextX]

    X_trainDFX = pd.DataFrame(X_trainX, index=None)
    X_testDFX = pd.DataFrame(X_testX, index=None)

    selected_featuresX = []
    accuracyX = []
    accuracyValX = np.array(accuracyX)
    selected_features_trainX = X_trainDFX.iloc[:, :feat]
    selected_features_testX = X_testDFX.iloc[:, :feat]
    train_setX = np.array(selected_features_trainX)
    test_setX = np.array(selected_features_testX)

    knnX = KNeighborsClassifier(n_neighbors=neigh)
    knnX.fit(train_setX, Y_trainX)

    predictionX = knnX.predict(test_setX)
    accuracyX.append(knnX.score(test_setX, Y_testX))
    c_mat = confusion_matrix(Y_testX,predictionX)
    perfMetrics(c_mat)

perf_Comp(chosen_K,bestFeat)