from __future__ import division
from math import *
import random
import numpy as np
from sklearn import svm

data = np.loadtxt("GalaxyZoo1_DR_table2.txt", delimiter = ",", skiprows = 1, usecols = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
labels = np.loadtxt("GalaxyZoo1_DR_table2.txt", delimiter = ",", skiprows = 1, usecols = (14,)) #only gets binary label col of elliptical

#IS IT NOT CLOSING THE FILE AFTER READING OR SOMETHING? HENCE ALTERNATING OUTPUTS? 

isElliptical = data[:,11] #corresponds to col 14 of real data file, which is the elliptical bool value
isSpiral = data[:,10]
isUncertain = data[:,12]

ellipticals = data[isElliptical == 1]

spirals = data[isSpiral == 1]
uncertains = data[isUncertain == 1]

trainingSetEllipticals = ellipticals[:500, :9] #check that these are first 500 and not last 500
trainingSetSpirals = spirals[:500, :9] #extracting first 500 spiral and elliptical to train model, excluding last 3 cols (labels)

trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))
print trainingSet.shape
trainingSetLabels = np.empty((len(trainingSetEllipticals) + len(trainingSetSpirals), ))

counter = 0
for label in trainingSetLabels:
    if counter < len(trainingSetEllipticals):
        trainingSetLabels[counter] = 1
    else:
        trainingSetLabels[counter] = 0

clf = svm.SVC(random_state = 3) #DON'T DO THIS!
#clf.fit(trainingSet, trainingSetLabels)
print clf.fit(trainingSet, trainingSetLabels)

#Training accuracy
numCorrect = 0
for i in range(len(trainingSetEllipticals)):
    prediction = clf.predict(trainingSetEllipticals[i])
    #print prediction
    if prediction[0] > 0.5:
        numCorrect += 1

print "Training accuracy = ", numCorrect/len(trainingSetEllipticals)


#Testing accuracy
testingSetElliptical = ellipticals[500:1000, :9]

numCorrect = 0
for i in range(len(testingSetElliptical)):
    prediction = clf.predict(testingSetElliptical[i])
    #print prediction
    if prediction[0] > 0.5:
        numCorrect += 1

print "Testing accuracy = ", numCorrect/len(testingSetElliptical)
