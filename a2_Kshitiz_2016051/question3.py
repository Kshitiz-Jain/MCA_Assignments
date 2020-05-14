#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# In[ ]:


file = open('trainXMFCC.pickle', 'rb')
train_X = pickle.load(file)
print(np.shape(train_X))

file2 = open('trainYMFCC.pickle', 'rb')
train_y = pickle.load(file2)
print(np.shape(train_y))

file3 = open('validationXMFCC.pickle', 'rb')
test_X = pickle.load(file3)
print(np.shape(test_X))

file4 = open('validationYMFCC.pickle', 'rb')
test_y = pickle.load(file4)
print(np.shape(test_y))

file5 = open('noiseXMFCC.pickle', 'rb')
noise = pickle.load(file5)
print(np.shape(noise))


# In[ ]:


file = open('trainXSPEC.pickle', 'rb')
train_X = pickle.load(file)
print(np.shape(train_X))

file2 = open('trainYSPEC.pickle', 'rb')
train_y = pickle.load(file2)
print(np.shape(train_y))

file3 = open('validationXSPEC.pickle', 'rb')
test_X = pickle.load(file3)
print(np.shape(test_X))

file4 = open('validationYSPEC.pickle', 'rb')
test_y = pickle.load(file4)
print(np.shape(test_y))

file5 = open('noiseSPEC.pickle', 'rb')
noise = pickle.load(file5)
print(np.shape(noise))


# In[ ]:


import random

#ToAddNoise
AMOUNT=0.5

row = np.shape(train_X)[0]
for i in range(row):
    rand = random.random()
    if(rand<0.5):
        totNum = np.shape(noise)[0]
        n = noise[random.randint(0,np.shape(noise)[0]-1)]
        d = train_X[i]
        r,c = np.shape(d)
        train_X[i] = (1-AMOUNT)*d + AMOUNT*n[:r,:c]
        

row = np.shape(test_X)[0]
for i in range(row):
    rand = random.random()
    if(rand<0.5):
        totNum = np.shape(noise)[0]
        n = noise[random.randint(0,np.shape(noise)[0]-1)]
        d = test_X[i]
        r,c = np.shape(d)
        test_X[i] = (1-AMOUNT)*d + AMOUNT*n[:r,:c]




trainX = []
for i in train_X:
    array =np.ravel(i)
    trainX.append(array)
print(np.shape(trainX))
print(np.shape(train_y))

testX = []
for i in test_X:
    array =np.ravel(i)
    testX.append(array)
print(np.shape(testX))
print(np.shape(test_y))


# In[ ]:


def minMaxScale(matrix):
    minimum = np.min(matrix,axis=0)
    maximum = np.max(matrix,axis=0)
    denom = maximum - minimum
    matrix = (matrix-minimum)/denom
    return matrix

trainX = minMaxScale(trainX)
testX = minMaxScale(testX)

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(trainX, train_y)
clf.score(testX,test_y)


# In[ ]:


with open('specNoiseClassifier.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[ ]:



def evaluations(pred,target):
    confMat = np.zeros((10,4))
    for i in range(10):
        #[TP,FP,FN,TN]
        pos=i
        length = np.shape(pred)[0]
        for j in range(length):
            if(target[j] == pos):
                if(pred[j] == target[j]):
                    confMat[i][0] += 1
                else:
                    confMat[i][2] +=1
            else:
                if(pred[j] != pos):
                    confMat[i][3] += 1
                else:
                    confMat[i][1] +=1
                    
    print(confMat)
    pres = confMat[:,0]/(confMat[:,0]+confMat[:,1])
    recall = confMat[:,0]/(confMat[:,0]+confMat[:,2])
    f1Score = 2*pres*recall/(pres+recall)
    print("p",pres)
    print("r",recall)
    print("f1",f1Score)
    print(np.average(pres),np.average(recall),np.average(f1Score))
    
    
    
# file = open('mfccClassifier.pkl', 'rb')
# clf = pickle.load(file)

# file = open('mfccNoiseClassifier.pkl', 'rb')
# clf = pickle.load(file)

file = open('specClassifier.pkl', 'rb')
clf = pickle.load(file)

# file = open('specNoiseClassifier.pkl', 'rb')
# clf = pickle.load(file)

print(clf.score(testX,test_y))
yPred = clf.predict(testX)

evaluations(yPred,test_y)


# In[ ]:




