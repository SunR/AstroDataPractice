from __future__ import division
from math import *
import random
import numpy as np
import astroML


#Learning a support vector machine, very popular type of linear classifier
#Just using binary-labeled 2-D vectors for now, [x1, y1] --> -1 (or 1)
#Use f = ax + by + c, and x and y are fixed inputs, a, b, c are "improved" by tugging (backprop)

step = 1e-2

class multiplicationGate (object):

    def __init__(self, unit1, unit2):
        self.unit1 = unit1
        self.unit2 = unit2 #check to see if this is actually necessary in python (comment out)
        
    def forward (self): #check if all arguments are necessary
        self.unitAbove = Unit(self.unit1.value * self.unit2.value, 0.0) #is the "self" necessary?
        return self.unitAbove
        
    def backward (self):
        #computing derivatives for this specific case, where dx=y*dt and dy=x*dt
        self.unit1.gradient = self.unit2.value * self.unitAbove.gradient 
        self.unit2.gradient = self.unit1.value * self.unitAbove.gradient

class additionGate (object):
    def __init__ (self, unit1, unit2):
        self.unit1 = unit1
        self.unit2 = unit2
        
    def forward(self):
        self.unitAbove = Unit (self.unit1.value + self. unit2.value, 0.0) #gradient values will be calculated in the backprop part
        return self.unitAbove
    
    def backward(self):
        self.unit1.gradient = 1.0 * self.unitAbove.gradient #derivative of addition function is just 1
        self.unit2.gradient = 1.0 * self.unitAbove.gradient
        
class Unit (object):
    def __init__(self, value, gradient):
        self.value = value  #value from forward pass
        self.gradient = gradient #gradient from backward pass

#just automates the single neuron process of tugging
class Circuit (object):
    def __init__(self, a, b, c, x, y):
        self.a = a
        self.b = b
        self.c = c
        self.x = x
        self.y = y
        
    def forward(self):
        self.multiplication1 = multiplicationGate(self.a, self.x)
        self.multiplication2 = multiplicationGate(self.b, self.y)
        self.ax = self.multiplication1.forward()
        self.by = self.multiplication2.forward()
        self.addition1 = additionGate(self.ax, self.by)
        self.axplusby = self.addition1.forward()
        self.addition2 = additionGate(self.axplusby, self.c)
        self.f = self.addition2.forward()
        return self.f
        
    def backward(self, gradientTop):
        self.f.gradient = gradientTop
        self.addition2.backward()
        self.addition1.backward()
        self.multiplication2.backward()
        self.multiplication1.backward()
        
class SVM (object):
    def __init__(self):
        self.a = Unit(1.0, 0.0) #change these to truly random starting conditions eventually!!
        self.b = Unit(-2.0, 0.0)
        self.c = Unit(-1.0, 0.0)
        
    def forward(self, x, y): #inputs, but in the form of Units this time
        self.x = x
        self.y = y
        self.circuit = Circuit(self.a, self.b, self.c, self.x, self.y)
        self.unitOutput = self.circuit.forward ()
        return self.unitOutput #first guess after 1 tug (?)
    
    #seeing if guess matched label, if not we tug
    def backward(self, label):
        self.tug = 0.0
        if label == -1 and self.unitOutput.value > -1: #too high, pull down
            self.tug = -1.0
        if label == 1 and self.unitOutput.value < 1: #too low, pull up
            self.tug = 1.0
        self.circuit.backward(self.tug)

    def updateInputs(self):
        step = 0.01
        self.a.value = self.a.value + step * self.a.gradient
        self.b.value = self.b.value + step * self.b.gradient
        self.c.value = self.c.value + step * self.c.gradient

    #run through entire learning iteration
    def learnIteration (self, x, y, label):
        self.forward(x, y)
        self.backward(label)
        self.updateInputs()
        
#Now train SVM with Stochastic Gradient Descent
#(linear classifier technique, simply adjusting a, b, c given inputs x, y)

##N = 6 #number of input vectors
##D = 2 #dimension of input vectors


#This can be done by reading an input file
data = np.loadtxt("GalaxyZoo1_DR_table2.txt", delimiter = ",", skiprows = 1, usecols = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
labels = np.loadtxt("GalaxyZoo1_DR_table2.txt", delimiter = ",", skiprows = 1, usecols = (14,)) #only gets binary label col of elliptical

#top line with names of cols skipped by loadtxt:
#OBJID,RA,DEC,NVOTE,P_EL,P_CW,P_ACW,P_EDGE,P_DK,P_MG,P_CS,P_EL_DEBIASED,P_CS_DEBIASED,SPIRAL,ELLIPTICAL,UNCERTAIN

for label in labels:
    if label==0:
        label = -1

#also should work: ellipticals = data[data[:, 12] == 1]
isElliptical = data[:,11] #corresponds to col 14 of real data file, which is the elliptical bool value
isSpiral = data[:,10]
isUncertain = data[:,12]

ellipticals = data[isElliptical == 1]
print ellipticals[:7, :] #print first 7 rows and all cols

spirals = data[isSpiral == 1]
uncertains = data[isUncertain == 1]

print len(ellipticals), len(spirals), len(uncertains), len(ellipticals) + len(spirals) + len(uncertains)

trainingSetEllipticals = ellipticals[:500, :] #check that these are first 500 and not last 500
trainingSetSpirals = spirals[:500, :] #extracting first 500 spiral and elliptical to train model

trainingSet = np.vstack((trainingSetEllipticals, trainingSetSpirals))
trainingSetLabels = labels

print trainingSet[:2, :]

for i in range(len(trainingSetEllipticals) + len(trainingSetSpirals)):
    if i <= len(trainingSetEllipticals):
        trainingSetLabels[i] = 1
    else:
        trainingSetLabels[i] = -1

data = trainingSet

print len (trainingSetEllipticals), len(trainingSetSpirals)

svm = SVM()

#runs through all data and checks if svm correctly predits each label
def evaluateTrainingAccuracy():
    numCorrect = 0
    for i in range (len(data)):
        x = data[i, 0]
        y = data[i, 1]
        x = Unit(x, 0.0)
        y = Unit(y, 0.0)
        unitOutput = svm.forward(x, y)
        trueLabel = trainingLabels[i,]
        predictedLabel = 1 if unitOutput.value > 0 else -1 #predicted label guessed by svm
        if predictedLabel == trueLabel:
            numCorrect += 1
    return numCorrect/len(data)

#the actual learning loop
#I STILL THINK THIS STARTS OUT TOO ACCURATE, CHECK EXPECTED OUTPUT AGAIN LATER
for iter in range(1000): 
    i = int(random.randrange(len(data)))
    x = data[i, 0]
    y = data[i, 1]
    label = trainingLabels[i,]
    svm.learnIteration(Unit(x, 0.0), Unit(y, 0.0), label)

    if iter % 25 == 0: #every 10 iterations print accuracy
        print "training accuracy at iteration", iter, evaluateTrainingAccuracy()

        
