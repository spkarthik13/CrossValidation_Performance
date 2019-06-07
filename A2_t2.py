import pandas as pd
import numpy as np
import random
import sys
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataSet = pd.read_csv(sys.argv[1], delimiter="\t", header=None) #Importing the dataset
dataSet.astype(float).values.tolist() #Converting it to float.

data = dataSet.iloc[:, :-1]
classification = dataSet.iloc[:, -1]

################################################################## Spearman correlation for feature selection #########################################################################
correlation = data.corr()
columns = np.full((correlation.shape[0],), True, dtype=bool)
for i in range(correlation.shape[0]):
    for j in range(i + 1, correlation.shape[0]):
        if correlation.iloc[i, j] >= 0.4: #Features below this threshold value are eliminated
            if columns[j]:
                columns[j] = False

columns_Selected = data.columns[columns]
data = data[columns_Selected]
datafinal = pd.DataFrame(data)
#########################################################################################################################################################################################
# print(len(selected_columns))
datafinal.insert(loc=len(columns_Selected), column="class", value=classification) # Final dataset with least correlated features

finalData = datafinal.astype(float).values.tolist() #Converting the dataset values to float.
random.shuffle(finalData) #Shuffling the dataSet to have mixed response variables.
finalDF = pd.DataFrame(finalData)

X = finalDF.iloc[:, :-1] #Dataset with just the features
X.astype(float).values.tolist()
Y = finalDF.iloc[:, -1] #Dataset with just the response variable
Y.astype(float).values.tolist()

def perfMetrics(mat):
    n = len(mat)
    di = mat.diagonal(0)
    r = pd.Index(["C1", "C2"])
    c = pd.Index(["P", "R", "Sp", "FDR"])

    confMat = pd.DataFrame(mat, index=r, columns=r)
    RS = []

    #false positive
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

    # true negative
    tn = []
    sumofmat = mat.sum()
    for i in range(n):
        rowSum = np.sum(mat[i])
        calSum = np.sum(mat[:, i]) - di[i]
        truenegative = sumofmat - (rowSum + calSum)
        tn.append(truenegative)

    # Precision
    P = []
    for i in range(n):
        precision = di[i] / RS[i]
        P.append(precision)

    # Accuracy
    TNTP = np.sum(di)
    PN = np.sum(mat)
    Accuracy = TNTP / PN

    #Recall
    rcl = []
    for i in range(n):
        TPFN = di[i] + fn[i]
        recal = di[i] / TPFN
        rcl.append(recal)

    #Specificity
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

################################################################ K- FoldCross validation #####################################################################################
def cross_validation(k):
    init = 0
    folds = 5
    next = int(len(finalData) / folds)
    totNext = int(len(finalData) / folds)

    featuresArray = []
    accuracyArray = []
    # print("---------------------------------------------------------------------")
    # print("Number of neighbors: ", k)
    # print("---------------------------------------------------------------------")
    for j in range(folds):

        X_train = pd.concat([X[:init], X[next:]]) #Training data containing features of instances other than the fold
        Y_train = pd.concat([Y[:init], Y[next:]]) #Training data containing response values of instances other than the fold
        # print("init : ", init, "next : ", next)
        X_test = X[init:next] #Test data containing features of instances in 1 fold
        Y_test = Y[init:next] #Test data containing response values of instances in 1 fold

        init = next
        next = next + totNext

        X_trainDF = pd.DataFrame(X_train, index=None)
        X_testDF = pd.DataFrame(X_test, index=None)
        colLen = X_trainDF.shape[1] # get the number of columns(features) in the data frame

        selected_features = []
        index = 1
        accuracy = [] #Array containing accuracies of selected features
        accuracyVal = np.array(accuracy)
        for i in range(colLen):
            selected_features_train = X_trainDF.iloc[:, :index+1]
            selected_features_test = X_testDF.iloc[:, :index+1]
            index += 1

            train_set = np.array(selected_features_train)
            test_set = np.array(selected_features_test)

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(train_set, Y_train) #Fitting the data to the knn classifier and training

            prediction = knn.predict(test_set)
            accuracy.append(knn.score(test_set, Y_test))
        maxAccFeatures = np.argmax(accuracy) + 1 #Index of the element having maximum accuracy in the array to find the number of features. Index corresponds to the number of selected features + 1(since index starts from 0)
        maxAcc = np.amax(accuracy) #Element with maximum accuracy in the array
        featuresArray.append(maxAccFeatures) #Array containing the number of features with maximum accuracies.
        accuracyArray.append(maxAcc) #Array containing maximum accuracies for the length of feature space. In our case 21 features are used
    # print("features with max accuracy: ",featuresArray)
    # print("Max accuracy in folds: ",accuracyArray)

    maxBestFeatind = np.argmax(accuracyArray) #Index of element containing maximum accuracy in the number of folds. In our case 5 folds are used, hence max accuracy among five folds.
    maxBestAcc = np.amax(accuracyArray)
    maxBestFeat = np.take(featuresArray,maxBestFeatind) #Get the best number of features containing maximum accuracy.

    #Store both the best accuracies and best features in an array for all the folds.
    overallAccuracy.append(maxBestAcc)
    overallFeatureSize.append(maxBestFeat)

    # print("Selected feature: ", maxBestFeat)


testrange = 5 # Testing with 5 different K values
K_Value = 3 # K value starting with 3

for test in range(testrange):
    cross_validation(K_Value)
    K_Value+=3 #Incrementing the K value by 3

bestAcc = np.amax(overallAccuracy)
bestFeat = np.take(overallFeatureSize,np.argmax(overallAccuracy)) #Getting the best feature size out of different K values
chosen_K = (np.argmax(overallAccuracy) +1) *3 #Get the number of neighbors (K in KNN) producing maximum accuracy
# print("overall feature sizes selected :", overallFeatureSize)
# print("overall accuracy: ", overallAccuracy)
# print("Best acc: ", bestAcc)
print("Features selected: ", bestFeat)
print("Neighbors selected: ",chosen_K)
print("-------------------------Performance metrics-------------------------------")

################################################## Performance metrics of chosen model ##############################################################################

def perf_Comp(neigh,feat): #Passing the chosen number of neighbors and feature size
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
    selected_features_trainX = X_trainDFX.iloc[:, :feat+1]
    selected_features_testX = X_testDFX.iloc[:, :feat+1]
    train_setX = np.array(selected_features_trainX)
    test_setX = np.array(selected_features_testX)

    knnX = KNeighborsClassifier(n_neighbors=neigh)
    knnX.fit(train_setX, Y_trainX)

    predictionX = knnX.predict(test_setX)
    accuracyX.append(knnX.score(test_setX, Y_testX))
    c_mat = confusion_matrix(Y_testX,predictionX)
    perfMetrics(c_mat)

perf_Comp(chosen_K,bestFeat)