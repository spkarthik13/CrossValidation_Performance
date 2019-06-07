import pandas as pd
import numpy as np
import random
import warnings
from collections import Counter

def k_nearest_neigbbors(data, predict, k):
    if len(data) >=k:
        warnings.warn('K is set to a value less than a total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append(([euclidian_distance, group]))

    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    # confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result


dataSet = pd.read_csv("A2_t2_dataset.tsv", delimiter="\t", header=None)
dataSet.astype(float).values.tolist()

data = dataSet.iloc[:, :-1]
classification = dataSet.iloc[:,-1]

correlation = data.corr()
columns = np.full((correlation.shape[0],), True, dtype=bool)
for i in range(correlation.shape[0]):
    for j in range(i+1, correlation.shape[0]):
        if correlation.iloc[i,j] >= 0.45:
            if columns[j]:
                columns[j] = False

selected_columns = data.columns[columns]
data = data[selected_columns]
datafinal = pd.DataFrame(data)

datafinal.insert(loc=len(selected_columns), column="class", value=classification)

finalData = datafinal.astype(float).values.tolist()
random.shuffle(finalData)
#finalDF = pd.DataFrame(finalData)
# print(datafinal)
finalDF = np.array(finalData)
#print(finalDF.shape)
#print(finalDF)

init = 0
folds = 5
next = int(len(finalData) / folds)
totNext = next
for _ in range(folds):
    train_set = {1:[], 0:[]}
    test_set = {1:[], 0:[]}
    train_data = np.delete(finalData, [init,next] , 0)
    test_data = finalData[init:next]
    print("Init: " +str(init), "Next: " +str(next))
    init = next +1
    next = next + totNext


    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct =0
    total =0

    for group in test_set:
        for data in test_set[group]:
            vote = k_nearest_neigbbors(train_set, data, k=5)
            # print(vote)
            if group == vote:
               correct +=1
            total+=1

    print ("Accuracy: ", correct/total)