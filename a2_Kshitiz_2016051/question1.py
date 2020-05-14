#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.io.wavfile as wav
import glob
from python_speech_features import sigproc
from scipy.fftpack import dct
import time

HZ=16000
WINDOW=int(25*HZ*0.001)
SAMPLERATE=int(10*HZ*0.001)
PREEMPH=0.97
LABEL = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"zero":0}

def applyHamming():
    alpha = 0.46164
    length = WINDOW
    fil = np.zeros(length)
    for i in range(length):
        fil[i]=((1-alpha)-alpha*np.cos([(2*np.pi*i)/(length-1)]))
    return fil

hammFilter = applyHamming()

def sig2Frames(sig):
    frames = []
    length = np.shape(sig)[0]
    for i in range(0,length-WINDOW,SAMPLERATE):
        signal = hammFilter*sig[i:i+WINDOW]
#         signal = applyHamming(sig[i:i+WINDOW])
        frames.append(signal)
    return frames


 


# In[ ]:


#Validation and training data

start = time.time()
# trainPath = glob.glob("./training/*")
trainPath = glob.glob("./validation/*")
trainPath.sort()
trainData = []
trainLabels = []
MAXFRAMES = 0
for i in trainPath:
    print(i)
#     label = LABEL[i[11:]]
    label = LABEL[i[13:]]
    cl = glob.glob(i+"/*.wav")
    for j in range(len(cl)):
        (rate,sig) = wav.read(cl[j])
        frame = sig2Frames(sig)
        MAXFRAMES = max(MAXFRAMES,np.shape(frame)[0])
        trainData.append(frame)
        trainLabels.append(label)
print("time taken :", time.time()-start)


# In[ ]:


#NOISE data
start = time.time()
trainPath = glob.glob("./_background_noise_/*")
trainData = []
MAXFRAMES = 0
for i in range(len(trainPath)):
    print(i)
    (rate,sig) = wav.read(trainPath[i])
    frame = sig2Frames(sig)
    print(np.shape(frame))
    MAXFRAMES = max(MAXFRAMES,np.shape(frame)[0])
    trainData.append(frame)
print("time taken :", time.time()-start)


# In[ ]:


print(np.shape(trainData))
print(np.shape(trainData[0]))
print(MAXFRAMES)


# In[ ]:


print(np.shape(trainData))
print(np.shape(trainData[0]))

def dftFilter():
    linArr = np.arange(WINDOW)
    linArr2 = np.reshape(linArr,(WINDOW,1))
    fil = np.exp( (-2j*np.pi*linArr*linArr2)/WINDOW )
    print(np.shape(fil))
    return fil


# def FFT(x):
#     x = np.asarray(x, dtype=float)
#     check = WINDOW % 2
#     if check != 0:
#         print("error")
#         continue
#     X_even = FFT(x[::2])
#     X_odd = FFT(x[1::2])
#     factor = np.exp(-2j * np.pi * np.arange(N) / N)
#     return np.concatenate([X_even + factor[:N / 2] * X_odd, X_even + factor[N / 2:] * X_odd])
    

fftFil = dftFilter()
row = np.shape(trainData)[0]
features = []
padd = np.zeros((1,13))
for i in range(row):
    sound= np.pad(trainData[i], ((0,MAXFRAMES-np.shape(trainData[i])[0]),(0,0)),'constant' )  
    row, col = np.shape(sound)
    for i in range(row):
          sound[i]=np.dot(fftFil,sound[i])
    fftSound = (1/WINDOW)*(np.abs(sound)**2)[:,:int(WINDOW/2) + 1]
#     fftSound = (1/WINDOW)*(np.abs(np.fft.rfft(sound,WINDOW))**2)
    features.append(fftSound)

print(np.shape(features))


# Reference Source : https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/


# In[ ]:


import pickle

with open('validationXSPEC.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('validationYSPEC.pickle', 'wb') as handle:
    pickle.dump(trainLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


import pickle

with open('trainXSPEC.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('trainYSPEC.pickle', 'wb') as handle:
    pickle.dump(trainLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


import pickle

with open('noiseSPEC.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

