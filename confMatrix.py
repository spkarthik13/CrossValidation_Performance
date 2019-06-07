import numpy as np
import pandas as pd
import sys

confusionMatrix = pd.read_csv(sys.argv[1], delimiter="\t", header = None)
intmat = confusionMatrix.astype(int).values.tolist()

def perfMetrics():
    mat = np.array(intmat)
    n = len(mat)
    di = mat.diagonal(0)
    r = pd.Index(["C1", "C2"])
    c = pd.Index(["P", "R", "Sp", "FDR"])

    confMat = pd.DataFrame(mat, index=r, columns=r)
    print(mat)
    RS = []
    for i in mat:
        sum = np.sum(i)
        RS.append(sum)

    fp = RS-di
    for k in fp:
        print(k)

    # false negative
    CS = []
    for i in range(n):
         sum = np.sum(mat[:,i])
         CS.append(sum)
    print("---------------------------------------------------")
    print("false negative:\n")

    fn = CS-di
    for k in fn:
        print(k)

    #true negative
    print("---------------------------------------------------")
    print("true negative:\n")
    tn = []
    sumofmat = mat.sum()
    for i in range(n):
        rowSum = np.sum(mat[i])
        calSum = np.sum(mat[:,i]) - di[i]
        truenegative = sumofmat - (rowSum + calSum)
        print(truenegative)
        tn.append(truenegative)
    print(tn)

    print("---------------------------------------------------")
    # precision
    print("precision:\n")
    P = []
    for i in range(n):
        precision = di[i]/RS[i]
        P.append(precision)
    print(P)

    print("---------------------------------------------------")
    #Accuracy
    TNTP = np.sum(di)
    PN = np.sum(mat)
    Accuracy = TNTP / PN
    print("Accuracy:\n")
    print(Accuracy)

    print("---------------------------------------------------")
    #Recal
    rcl = []
    print("Recal:\n")
    for i in range(n):
        TPFN = di[i] + fn[i]
        recal = di[i] / TPFN
        rcl.append(recal)
    print(rcl)

    print("---------------------------------------------------")
    #specivity
    spcty = []
    print("specivity:\n")
    for i in range(n):
        FPTN = fp[i] + tn[i]
        specificity = tn[i] / FPTN
        spcty.append(specificity)
    print(spcty)

    print("---------------------------------------------------")
    #FDR
    FDR = []
    print("FDR:\n")
    for i in range(n):
        FPTP = fp[i] + di[i]
        falseDisR = fp[i] / FPTP
        FDR.append(falseDisR)
    print(FDR)

    print("-----------------------------------------------------")
    print("Confusion Matrix","\n")
    print(confMat,"\n\n")

    print("Ac: " + str(Accuracy),"\n")
    print("Performance metrics: \n")
    perfMat = np.array([P,rcl,spcty,FDR]).transpose()
    perfDataFrame = pd.DataFrame(perfMat, index=r, columns=c)
    print(perfDataFrame)
