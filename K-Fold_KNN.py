import pandas as pd
import numpy as np
import random
import warnings
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
    maxBestFeat = np.take(featuresArray,maxBestFeatind)

    print("Selected feature: ", maxBestFeat)
    #     print("Fold: ", j+1)
    #     print("Best feature size: ",maxAccFeatures) #Returns the indexd value of maximum accuracy which describes the number of features used.
    #     print("Best accuracy: ", maxAcc) #Returns the maximum accuracy
    #     # print("Avg accuracy : ", np.sum(accuracy) / len(accuracy))
    #     print("-------------------------------------")
    #
    # print("Average feature size for best accuracy: ",np.sum(avgFeatures)/len(avgFeatures))
    # print("Average accuracy: ", np.sum(avgAccuracy)/len(avgAccuracy))
    # print("\n")
    # overallAccuracy.append(np.sum(avgAccuracy)/len(avgAccuracy))
    # overallFeatureSize.append(np.sum(avgFeatures)/len(avgFeatures))

cross_validation(15)

# testrange = 5
# K_Value = 3
# for test in range(testrange):
#     cross_validation(K_Value)
#     K_Value+=3
#
# bestAcc = np.amax(overallAccuracy)
# bestFeat = np.take(overallFeatureSize,np.argmax(overallAccuracy))
# chosen_K = (np.argmax(overallAccuracy) +1) *3
# print("overall feature size :", overallFeatureSize)
# print("overall accuracy: ", overallAccuracy)
# print("Best acc: ", bestAcc)
# print("Best feature size: ", bestFeat)
# print("Neighbors: ",chosen_K)