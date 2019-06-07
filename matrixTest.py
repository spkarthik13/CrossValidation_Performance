import numpy as np
import pandas as pd
import sys

confusionMatrix = pd.read_csv(sys.argv[1], delimiter="\t", header = None)
intmat = confusionMatrix.astype(int).values.tolist()
mat = np.array(intmat)
n = len(mat)
di = mat.diagonal(0)
r = pd.Index(["C1", "C2", "C3"])
c = pd.Index(["P", "R", "Sp", "FDR"])

confMat = pd.DataFrame(mat, index=r, columns=r)
#false positive
RS = []
for i in mat:
    sum = np.sum(i)
    RS.append(sum)

fp = RS-di

# false negative
CS = []
for i in range(n):
     sum = np.sum(mat[:,i])
     CS.append(sum)

fn = CS-di
#true negative
tn = []
sumofmat = mat.sum()
for i in range(n):
    rowSum = np.sum(mat[i])
    calSum = np.sum(mat[:,i]) - di[i]
    truenegative = sumofmat - (rowSum + calSum)
    tn.append(truenegative)

P = []
for i in range(n):
    precision = di[i]/RS[i]
    P.append(precision)

#Accuracy
TNTP = np.sum(di)
PN = np.sum(mat)
Accuracy = TNTP / PN
#Recal
rcl = []
for i in range(n):
    TPFN = di[i] + fn[i]
    recal = di[i] / TPFN
    rcl.append(recal)

#specivity
spcty = []
for i in range(n):
    FPTN = fp[i] + tn[i]
    specificity = tn[i] / FPTN
    spcty.append(specificity)

FDR = []
for i in range(n):
    FPTP = fp[i] + di[i]
    falseDisR = fp[i] / FPTP
    FDR.append(falseDisR)


print("Ac: " + str(Accuracy),"\n")
print("Performance metrics: \n")
perfMat = np.array([P,rcl,spcty,FDR]).transpose()
perfDataFrame = pd.DataFrame(perfMat, index=r, columns=c)
print(perfDataFrame)
