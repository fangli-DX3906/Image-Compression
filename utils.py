import numpy as np
import matplotlib.pyplot as plt


def assignToMean(data, meanVec):
    distMat = []
    for i in range(len(meanVec)):
        diff = data - np.tile(meanVec[i], (data.shape[0], 1))
        distsq = diff ** 2
        dist = np.sum(distsq, axis=1)
        distMat.append(dist)
    distMat = np.asarray(distMat)
    sortedIndex = np.argsort(distMat, axis=0)
    thisCluster = sortedIndex[0, :]
    return thisCluster


def meanCal(data, clstr, k):
    meanVec = []
    for i in range(k):
        meanVec.append(np.mean(data[clstr == i, :], axis=0))
    return meanVec


def KMNS(data, k):
    # initialize
    meanVec = []
    for i in range(k):
        point = 255 / (k - 1)
        meanVec.append(np.array([point * i, point * i, point * i]))

    error = 1e-3
    diff = 1
    count = 0
    while diff >= error:
        count += 1
        clusterIndex = assignToMean(data, meanVec)
        meanVecNew = meanCal(data, clusterIndex, k)
        diff = np.sum((np.array(meanVec) - np.array(meanVecNew)) ** 2)
        meanVec = meanVecNew

    return clusterIndex, meanVec, count


def reconstructImg(dimension, category, mean):
    img = np.zeros((category.shape[0], 3))
    for i in range(len(mean)):
        img[category == i, :] = mean[i]
    img = np.asarray(img)
    img = img.reshape(dimension)
    img = img.astype(int)

    return img


def plotSelect(meanVec):
    color = np.tile(meanVec[0], (300, 1)).reshape((20, 15, 3))
    for i in range(1, len(meanVec)):
        thiscolor = np.tile(meanVec[i], (300, 1)).reshape((20, 15, 3))
        color = np.concatenate((color, thiscolor), axis=1)
    color = color.astype(int)
    plt.imshow(color)
    plt.show()
