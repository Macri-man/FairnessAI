import numpy as np
import sys
from time import time
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris



class AdaBoost:
    def __init__(self,T, data, labels, classifier):
        self.T = T
        self.Y = self.FixLabels(labels)
        self.X = data
        self.m = len(self.Y)
        self.D = [1.0/self.m] * self.m
        self.clf = classifier
        self.Ht = [0] * self.T
        self.alphas = [0] * self.T
        self.et = [0] * self.T
        self.Error = []

    def FixLabels(self,labels):
        return [-1 if l % 2 else 1 for l in labels]

    def sign(self,xs):
        return [1 if x >= 0 else -1 for x in xs]

    def randomSample(self,sampleSize):
        indices = np.sort(np.random.choice(self.m,sampleSize,replace=False,p=self.D))
        return [self.D[i] for i in indices],self.X[indices],[self.Y[i] for i in indices]

    def getError(self,t):
        product = 1
        for i in range(t):
            product *= (2 * math.sqrt(self.et[i]*(1-self.et[i])))
        self.Error.append(product)

    def Train(self):
        percentage = 0.05
        sampleSize = int(self.m*percentage)
        for t in range(self.T):
            print "T %d" % t
            tempD,tempX,tempY = self.randomSample(sampleSize)
            self.clf.fit(tempX,tempY,sample_weight=tempD)
            self.Ht[t] = self.clf.predict(self.X)
            self.et[t] = sum(self.D * (self.Ht[t] != self.Y))
            self.alphas[t] = 0.5 * np.log((1-self.et[t])/self.et[t])
            zt = sum([self.D[i]*math.exp(-self.alphas[t]*self.Y[i]*self.Ht[t][i]) for i in range(self.m)])
            self.D = [(self.D[i]*math.exp(-self.alphas[t]*self.Y[i]*self.Ht[t][i]))/zt for i in range(self.m)]
            self.getError(t)
        return self.sign(sum([self.alphas[t]*self.Ht[t] for t in range(self.T)]))


if __name__ == "__main__":

    data,target = datasets.load_digits(return_X_y=True)

    datairis,targetiris = datasets.load_iris(return_X_y=True)