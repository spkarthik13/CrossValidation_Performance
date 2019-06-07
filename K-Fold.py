import pandas as pd
import numpy as np
import random
import warnings
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


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

X = finalDF.iloc[:,:-1]
X.astype(float).values.tolist()
Y = finalDF.iloc[:,-1]
Y.astype(float).values.tolist()
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

init = 0
folds = 5
next = int(len(finalData) / folds)
totNext = int(len(finalData) / folds)

for j in range(folds):

    X_train = pd.concat([X[:init] , X[next:]])
    Y_train = pd.concat([Y[:init], Y[next:]])
    print("init : ", init, "next : ",next)
    X_test = X[init:next]
    Y_test = Y[init:next]
    init = next
    next = next + totNext

    X_trainDF = pd.DataFrame(X_train,index=None)
    X_testDF = pd.DataFrame(X_test,index=None)
    colLen = X_trainDF.shape[1]

    selected_features = []
    index =1
    accuracy = []
    for i in range(colLen):
        selected_features_train = X_trainDF.iloc[:,:index]
        selected_features_test = X_testDF.iloc[:, :index]
        index+=1
        train_set = np.array(selected_features_train)
        test_set = np.array(selected_features_test)
        #print(train_set)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_set, Y_train)
        prediction = knn.predict(test_set)
        accuracy.append(knn.score(test_set, Y_test))
    print("Avg accuracy : ", np.sum(accuracy)/len(accuracy))
    print("-------------------------------------")
# print(X_trainDF)


# knn.predict(X_test,3)