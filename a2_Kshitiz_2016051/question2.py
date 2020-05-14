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

def preEmphasis(signal):
    newSignal = []
    row = np.shape(signal)[0]
    for i in range(1,row):
        val = signal[i] - PREEMPH*signal[i-1]
        newSignal.append(val)
    return np.array(newSignal)


 


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
        sig=preEmphasis(sig)
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
    sig=preEmphasis(sig)
    frame = sig2Frames(sig)
    print(np.shape(frame))
    MAXFRAMES = max(MAXFRAMES,np.shape(frame)[0])
    trainData.append(frame)
print("time taken :", time.time()-start)


# In[ ]:


print(np.shape(trainData))
print(np.shape(trainData[0]))

def getfb(nfilt=26,nfft=WINDOW):
    highfreq=HZ/2
    highm = 2595 * np.log10(1+highfreq/700.)
    lowm = 0
    
    melpts = np.linspace(lowm,highm,nfilt+2)
    hzpts = 700*(10**(melpts/2595.0)-1)
    
    bins = np.floor((nfft+1)*hzpts/HZ)
    
    freqbank = np.zeros((nfilt,nfft//2+1))
    for j in range(0,nfilt):
        a = bins[j]
        b = bins[j+1]
        c = bins[j+2]
        for i in range(int(a), int(b)):
            freqbank[j,i] = (i - a) / (b-a)
        for i in range(int(b), int(c)):
            freqbank[j,i] = (c-i) / (c-b)
    return freqbank


row = np.shape(trainData)[0]
fb = getfb()
features = []
padd = np.zeros((1,13))
for i in range(row):
    sound= np.pad(trainData[i], ((0,MAXFRAMES-np.shape(trainData[i])[0]),(0,0)),'constant' ) 
    energy = np.sum(np.square(sound),1)
    energy = np.reshape(energy, (np.shape(energy)[0],1))
    
    #CACLULATING FEATURES 
    fftSound = (1/WINDOW)*(np.abs(np.fft.rfft(sound,WINDOW))**2)
    f = np.dot(fftSound,fb.T)
    f = f + np.finfo(float).eps
    f=10*np.log(f)
    
    f = dct(f, axis=1)[:,1:13]
    
    array = np.arange(0, 12)
    CEP = 20
    lift = 1+(CEP/2)*np.sin(np.pi*array/CEP)
    f = f*lift
    f = f-np.mean(f)
    
    
    #ADDING ENERGY
    f = np.concatenate((f, energy), axis=1)
    
    #DELTA FEATURES
    f = np.concatenate((f, padd), axis=0)
    f = np.concatenate((padd, f), axis=0)
    
    row,col = np.shape(f)
    tempF = np.zeros(np.shape(f))
    for i in range(1,row-1):
        tempF[i] = (f[i+1]-f[i-1])/2
    
    tempF2 = np.zeros(np.shape(f))
    for i in range(1,row-1):
        tempF2[i] = (tempF[i+1]-tempF[i-1])/2
    
    f=np.concatenate((f,tempF),axis=1)
    f=np.concatenate((f,tempF2),axis=1)
    f = np.delete(f,0,0)
    f = np.delete(f,-1,0)
#     if(i%20==0):
#         print(i,np.shape(f))
    features.append(f)

print(np.shape(features))

#Reference source : https://github.com/jameslyons/python_speech_features


# In[ ]:


import pickle

with open('validationXMFCC.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('validationYMFCC.pickle', 'wb') as handle:
    pickle.dump(trainLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


import pickle

with open('trainXMFCC.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('trainYMFCC.pickle', 'wb') as handle:
    pickle.dump(trainLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


import pickle

with open('noiseXMFCC.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




