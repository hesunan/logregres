import numpy as np
from numpy import *


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMat,labelMat):
    dataMatrix = np.mat(dataMat)  #translate list to matrix
    labelMatrix = np.mat(labelMat).transpose() #转置
    m,n = np.shape(dataMatrix) #100 rows  3 coulums
    alpha = 0.001 #步长 or 学习率
    maxCyclse = 500
    weight = ones((n,1)) #初始值随机更好吧
    #weight = np.random.rand(n,1)
    for k in range(maxCyclse):
        h = sigmoid(dataMatrix * weight) # h 是向量
        error = (labelMatrix - h)  #error 向量
        weight = weight + alpha * dataMatrix.transpose() *error  #更新
        #print(k,"  ",weight)
    return weight

#垃圾python，51行显示x，y维度不一致，有毒吧
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
#这样，x，y维度就一致了。。。。
#import matplotlib.pyplot as plt
#dataMat,labelMat=loadDataSet()
#dataArr = array(dataMat)
#n = shape(dataArr)[0] 
#xcord1 = []; ycord1 = []
#xcord2 = []; ycord2 = []
#for i in range(n):
#    if int(labelMat[i])== 1:
#        xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
#    else:
#        xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
#ax.scatter(xcord2, ycord2, s=30, c='green')
#x = np.arange(-3.0, 3.0, 0.1)
#y=(-4.12414349-0.48007329*x)/(-0.6168482)
#ax.plot(x, y)
#plt.xlabel('X1'); plt.ylabel('X2');
#plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #初始化weights
    for j in range(numIter):
        dataIndex = list(range(m))#转换为列表后进行索引
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#随机下标生成
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#*******************预测病马的死亡率*****************************
#对回归系数进行分类
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

