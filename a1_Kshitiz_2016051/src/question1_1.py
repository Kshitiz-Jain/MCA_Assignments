from PIL import Image
import numpy as np
import glob
import pickle
from sklearn.cluster import KMeans
import cv2
import time
import heapq
# img=Image.open("./images/all_souls_000008.jpg")
# img2=img.quantize(16)
# array=np.array(img2)
# print(array.shape)
# print(array)


def getCluster():
	X=[]
	for i in range(16):
		for j in range(16):
			for k in range(16):
				X.append([i,j,k])
	kmeans = KMeans(n_clusters=16).fit(X)
	return kmeans


def getCorrelogram(image):
	dist=[1,3,5,7]
	di={1:0,3:1,5:2,7:3}
	features=np.zeros((16,4))
	totNeigh=np.ones((16,4))
	row,col = np.shape(image)
	for i in range(row):
		for j in range(col):
			c=image[i][j]
			visited=np.zeros((row,col))
			def findColor(curr,p,q):
				if(visited[p,q]==1):
					return
				visited[p][q]=1
				d = max(abs(p-i),abs(q-j))
				if(d>max(dist)):
					return
				if(d in dist):
					if(image[i][j]==image[p][q]):
						features[curr][di[d]]+=1
					totNeigh[curr][di[d]]+=1
				if(p>0):
					findColor(curr,p-1,q)
				if(q>0):
					findColor(curr,p,q-1)
				if(p<row-1):
					findColor(curr,p+1,q)
				if(q<col-1):
					findColor(curr,p,q+1)
			findColor(c,i,j)
	return np.divide(features,totNeigh)


def quantiseImg(image,cluster):
	image = image/16
	image= cv2.resize(image,(0,0),fx=0.1,fy=0.1)
	row,col,k = np.shape(image)
	newImg=np.zeros((row,col))
	points=image.reshape(row*col,3)
	newImg=cluster.predict(points)
	newImg=newImg.reshape(row,col)
	return newImg


def createFeatures():
	cluster=getCluster()

	totData={}
	paths = glob.glob("./images/*.jpg")
	for i in paths[:1]:
		key=i[9:len(i)-4]
		print(key)
		img=cv2.imread(i)
		img=quantiseImg(img,cluster)
		acc=getCorrelogram(img)
		totData[key]=acc

	with open('quanData16.pickle', 'wb') as handle:
		pickle.dump(totData, handle, protocol=pickle.HIGHEST_PROTOCOL)



def getQueryName():
	queries = []
	qNum=[]
	paths=glob.glob("./train/query/*_query.txt")
	paths.sort()
	count=0
	for i in paths:
		q=open(i,"r").readlines()
		q=q[0].rstrip("\n").split(" ")
		queries.append(q[0][5:])
		qNum.append(i[14:-10])
	return queries,qNum



def evaluate(heap,name):
	goodFile = open("./train/ground_truth/"+name+"_good.txt").readlines()
	goodFile = list(map(lambda s: s.strip(), goodFile))
	okFile = open("./train/ground_truth/"+name+"_ok.txt").readlines()
	okFile = list(map(lambda s: s.strip(), okFile))
	junkFile = open("./train/ground_truth/"+name+"_junk.txt").readlines()
	junkFile = list(map(lambda s: s.strip(), junkFile))
	array=[0,0,0]
	array2=[len(goodFile),len(okFile),len(junkFile)]
	for i in heap:
		curr= i[1]
		if(curr in goodFile):
			array[0]+=1
		if(curr in okFile):
			array[1]+=1
		if(curr in junkFile):
			array[2]+=1
	return array, array2
		

def getDistance(img1,img2):
	img3=np.absolute(img1-img2)
	img4=np.ones(np.shape(img1))+img1+img2
	return np.sum(img3/img4)/32


def results(array1,array2,checked):
	retrived = []
	total = []
	for i in range(len(array1)):
		retrived.append(np.sum(array1[i]))
		total.append(np.sum(array2[i]))

	perGood= np.average( np.divide(np.array(array1[:][0]),np.array(array2[:][0])) ) *100
	perOk= np.average( np.divide(np.array(array1[:][1]),np.array(array2[:][1])) ) *100
	perJunk= np.average( np.divide(np.array(array1[:][2]),np.array(array2[:][2])) ) *100

	print(perGood,perOk,perJunk)




	avgP = np.average(retrived)/checked
	minP = np.min(retrived)/checked
	maxP = np.max(retrived)/checked
	
	# print("Average Precision :", np.average(retrived)/checked)
	# print("Minimum Precision :", np.min(retrived)/checked)
	# print("Maximum Precision :", np.max(retrived)/checked)
	# print(avgP,minP,maxP)

	avgR = np.average(np.divide(retrived,total))
	minR = np.min(np.divide(retrived,total))
	maxR = np.max(np.divide(retrived,total))

	# print("Average Recall :", np.average(np.divide(retrived,total)))
	# print("Minimum Recall :", np.min(np.divide(retrived,total)))
	# print("Maximum Recall :", np.max(np.divide(retrived,total)))
	# print(avgR,minR,maxR)

	avgF1 = np.average( np.divide(2*(np.array(retrived)/checked)*np.divide(retrived,total), np.array(retrived)/checked+np.divide(retrived,total) ))
	minF1 = np.min( np.divide(2*(np.array(retrived)/checked)*np.divide(retrived,total), np.array(retrived)/checked+np.divide(retrived,total) ))
	maxF1 = np.max( np.divide(2*(np.array(retrived)/checked)*np.divide(retrived,total), np.array(retrived)/checked+np.divide(retrived,total) ))

	# avgF1 = (2 * avgP * avgR)/ (avgR+avgP)
	# minF1 = (2 * minP * minR)/ (minR+minP)
	# maxF1 = (2 * maxP * maxR)/ (maxR+maxP)

	# print(avgF1,minF1,maxF1)

	avgGood=np.average(array1[:][0])
	avgok=np.average(array1[:][1])
	avgjunk=np.average(array1[:][2])

	# print(avgGood,avgok,avgjunk)








def retriveImages():
	file = open("quanData16.pickle", 'rb')
	# file = open("corrDic.pkl", 'rb')
	data=pickle.load(file)
	query,qNum=getQueryName()
	array1=[]
	array2=[]
	timed=[]
	for i in range(len(query)):
		heap=[]
		# qImg=data[query[i]+".jpg"]
		start=time.time()
		qImg=data[query[i]]
		for j in data.keys():
			dataImg=data[j]
			dist = getDistance(dataImg,qImg)
			# heap.append([dist,j[:-4]])
			heap.append([dist,j])
		heap.sort(key = lambda x : x[0])
		heap=heap[:200]
		temp1, temp2 = evaluate(heap,qNum[i]),qNum[i]
		timed.append(time.time()-start)
		array1.append(temp1[0])
		array2.append(temp1[1])
	# print(np.average(timed))
	results(array1,array2,200)	


retriveImages()










