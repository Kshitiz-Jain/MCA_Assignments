import numpy as np
from collections import Counter
import nltk
import string
from nltk.corpus import stopwords
from nltk.corpus import abc
import copy
import re
from scipy.special import softmax
import pickle

# nltk.download('stopwords')

class NN():
	"""docstring for NN"""
	def __init__(self, N,V):
		super(NN, self).__init__()
		self.lr = 0.001
		self.inputSize=N
		self.hiddenSize=V
		self.start=10
		self.W1 = np.random.uniform(-0.9, 0.9, (self.inputSize, self.hiddenSize))
		# self.W2 = np.random.uniform(-0.9, 0.9, (self.hiddenSize, self.inputSize))
		# self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
		self.W2 = np.random.randn(self.hiddenSize, self.inputSize)
		# self.W1 = np.load("W1_"+str(self.start)+".npy")
		# self.W2 = np.load("W2_"+str(self.start)+".npy")
		
		

	def forward(self,X):
		X= np.array(X).reshape(self.inputSize,1)
		self.z1 = np.matmul(self.W1.T, X)
		self.z2 = np.matmul(self.W2.T, self.z1)
		
		#SOFTMAX
		# softMax = np.exp(self.z2-np.max(self.z2))
		# softMax = softMax/softMax.sum(axis=0) 
		soft = softmax(self.z2)
		return soft

	def backprop(self,target,error):
		target=np.array(target).reshape(self.inputSize,1)

		stepOne = np.matmul(self.z1,error.T)/self.inputSize
		stepTwo = np.matmul(self.W2,error)
		stepThree = np.matmul(target,stepTwo.T)		

		self.W1 += -1* self.lr*stepThree
		self.W2 += -1* self.lr*stepOne


	def loss(self, context):
		outLayer = self.z2
		temp = 0
		for j in context:
			temp -= outLayer[j]
		temp += len(context)*np.log(np.sum(np.exp(outLayer)))
		return temp


	def train(self,trainDataX,trainDataY,epochs):
		row,col = np.shape(trainDataY)
		for k in range(epochs):
			loss = 0
			for i in range(row):
				target = trainDataY[i]
				context = trainDataX[i]

				out = self.forward(target)
				
				numContext = len(context)
				error = copy.deepcopy(out*numContext)
				error[context,0] -=1

				self.backprop(target,error)
				
				b =-1 * np.sum(np.log(out[context,0] + 0.001))
				loss +=b
			print(loss)
			np.save("W1_"+str(k+self.start+1),self.W1)
			np.save("W2_"+str(k+self.start+1),self.W2)



class Word2Vec(object):
	"""docstring for Word2Vec"""
	def __init__(self,context):
		super(Word2Vec, self).__init__()
		self.contextSize = context
		self.nn = None

	def preprocessing(self,corpus): 
		stop_words = set(stopwords.words('english'))    
		# nums = ["0","1","2","3","4","5","6","7","8","9","0"] 
		nums = "0123456789"
		training_data = []
		for i in range(np.shape(corpus)[0]): 
			sentence = corpus[i]
			data = []
			for word in sentence:
				word =word.lower()
				if(word not in stop_words and len(word)>0):
					temp = re.sub(r"[,.;@#?!&$-0123456789()\"\']+", '', word)
					# temp = re.sub(r"0123456789", '', temp)
					# temp = word.strip(string.punctuation)
					# temp = word.strip(nums)
					if(len(temp)>0):
						data.append(temp)
			training_data.append(data)
		return training_data


	def hotEncodeWord(self,word):
		array = np.zeros(self.uniqueCount)
		array[self.word2IndexMap[word]]=1
		return list(array)

	def hotEncodeID(self,wordId):
		array = np.zeros(self.uniqueCount)
		array[wordId]=1
		return array

	def targetAndContext(self,dataset):
		word2IndexMap={}
		index2WordMap={}
		count=0
		for data in dataset:
			for i in range(len(data)):
				if(data[i] not in word2IndexMap):
					word2IndexMap[data[i]]=count
					index2WordMap[count]=data[i]
					count+=1
		self.word2IndexMap=word2IndexMap
		self.index2WordMap=index2WordMap
		self.uniqueCount=count
		print(count)
		trainDataX = []
		trainDataY = []
		for i in range(self.uniqueCount):
			trainDataY.append(self.hotEncodeID(i))
			trainDataX.append([])
		counter=0
		for data in dataset:
			for i,word in enumerate(data):
				contextWords = []
				for j in range(i-self.contextSize,i+1+self.contextSize):
					if(j==i):
						continue
					else:
						if(j>=0 and j<len(data)):
							contextWords.append(self.word2IndexMap[data[j]])
				trainDataX[self.word2IndexMap[word]] += contextWords
		for key in range(self.uniqueCount):
			trainDataX[key] = list(set(trainDataX[key]))
		return trainDataX, trainDataY


	def convert2Vec(self,word):
		index = self.word2IndexMap[word]
		return self.nn.W1[index]


	def findSimilarWords(self,word, count):
		targetVec = self.convert2Vec(word)
		simWords = []
		for i in range(self.uniqueCount):
			outVec = self.nn.W1[i]
			num = np.dot(targetVec,outVec)
			dem = np.linalg.norm(targetVec)*np.linalg.norm(outVec)
			distance = -1*num/dem
			simWords.append( (i,distance) )
		simWords.sort(key = lambda x: x[1])

		for i in range(count):
			idx = simWords[i][0]
			print(self.index2WordMap[idx])


def dumpPickle(filname, dic):
	file=open(filname,'wb')
	pickle.dump(dic,file)
	file.close()


def saveContextWords(data):
	context={}
	for i in range(len(data)):
		context[i] = data[i]
	dumpPickle("contextWords.pkl", context)



if __name__ == '__main__':
	obj = Word2Vec(2) 
	
	# corpus = "natural language processing and machine learning is fun and exciting".split(" ")
	# sentences = [corpus]

	sentences=list(abc.sents())

	data = obj.preprocessing(sentences)
	print("preprocessing done")
	trainDataX, trainDataY = obj.targetAndContext(data)
	print("training Data is generated")
	exit(0)
	dumpPickle("index2WordMap.pkl",obj.index2WordMap)
	dumpPickle("word2IndexMap.pkl",obj.word2IndexMap)
	saveContextWords(trainDataX)



	# network = NN(obj.uniqueCount, 50)
	# network.train(trainDataX,trainDataY,100)
	# obj.nn = network
	# obj.findSimilarWords("natural",3)
