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

class sets:
    def __init__(self,FairnessPolicies,Intersections):
        self.sets = FairnessPolicies.values()
        self.inter = Intersections
    def getSets(self):
        ssets = []
        ssets.append(self.sets[0])
        for i in self.sets[1:]:
            ssets[0] = ssets[0] - i
        for intersects in self.inter:
            ssets.append([])

class FairnessPolicy:
    def __init__(self,FairnessPolicies):
        self.FP = FairnessPolicies
        self.sets = self.getSets()
    def getFairnessOutcome(self,key):
        return self.FP[key]
    

class FairnessAlg:
    def __init__(self,FairnessOutcomeDesc,FairnessPolicies,data,target,MachineLearningAlg):
        self.FOD = FairnessOutcomeDesc
        self.FP = FairnessPolicies
        self.MLAG = MachineLearningAlg
    def determineFairness(self,MLAG):
        data = MLAG["data"]
        target = []
        MLA.fit(self.data, self.target)
        t = set(target)
        if self.MLA.score() == 0.0:
            for d in self.FP:
                if t in d["FOutcome"]:
                    d["FMLA"].append()


if __name__ == "__main__":

    FairnessOutcomeDesc = ["The leaf is going to burn","Flower cause is raiesd","The flower cost is decreased","The stem is to tall"]
    FairnessPolicies = [{"FName": 'A', "FOutcome": 1, 2, 4, "FMLA": []}, {"FName": 'B', "FOutcome": 1, 3, "FMLA": []}, {"FName": 'U', "FOutcome": 1, 2, 3, 4, "FMLA": []}]

    data,target = datasets.load_digits(return_X_y=True)
    datairis,targetiris = datasets.load_iris(return_X_y=True)
