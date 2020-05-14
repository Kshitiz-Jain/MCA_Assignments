import pickle
import numpy as np




def loadPicle(filename):
	file = open(filename,'rb')
	dic = pickle.load(file)
	file.close()
	return dic




def findSimilarWords(word, count, index2WordMap, word2IndexMap, W):
	targetVec = W[ word2IndexMap[word] ] 
	simWords = []
	for i in range(W.shape[0]):
		outVec = W[i]
		num = np.dot(targetVec,outVec)
		dem = np.linalg.norm(targetVec)*np.linalg.norm(outVec)
		distance = -1*num/dem
		simWords.append( (i,distance) )
	simWords.sort(key = lambda x: x[1])

	for i in range(count):
		idx = simWords[i][0]
		print(index2WordMap[idx])



weigths = np.load("./NN/W1_13.npy")
index2WordMap = loadPicle("./NN/index2WordMap.pkl")
word2IndexMap = loadPicle("./NN/word2IndexMap.pkl")

findSimilarWords("fossil", 10, index2WordMap, word2IndexMap, weigths)
