import numpy as np
import cv2
import glob
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import time
import pywt
import pickle
import heapq
import json
from skimage.feature import blob
from skimage.feature import hessian_matrix_det


def getQuery():
	queries = {}
	paths=glob.glob("./train/query/*query.txt")
	paths.sort()
	count=0
	for i in paths:
		q=open(i,"r").readlines()
		q=q[0].rstrip("\n").split(" ")
		image=cv2.imread("./images/"+q[0][5:]+".jpg",0)
		queries[q[0][5:]]=image
	return queries


def padding(image,padd):
	psize=math.floor(padd/2)
	row,col = image.shape
	nimg=np.zeros((row+2*psize,col+2*psize))
	for i in range(row):
		for j in range(col):
			nimg[i+psize][j+psize]=image[i][j]
	return nimg


def removepadd(image, ker):
	row,col=image.shape
	pad=int(math.floor(ker/2))
	newimg=np.zeros((row-2*pad,col-2*pad))
	for i in range(row-2*pad):
		for j in range(col-2*pad):
			newimg[i][j]=image[i+pad][j+pad]
	return newimg



def getSURFFeatures2(image, threshold, levels):
	image = cv2.GaussianBlur(image,(3,3),0)
	images = []
	zero0=np.zeros(np.shape(image))
	zero1=np.zeros(np.shape(image))
	# zero2=removepadd(zero1,40)
	# zero3=removepadd(zero1,40)
	images.append(zero1)
	maxs=[0]
	for i in range(1,levels):
		temp=hessian_matrix_det(image, sigma=i)
		# temp=removepadd(temp,40)
		cv2.imwrite(str(i)+".jpg",temp*255)
		images.append(temp)
		maxs.append(np.max(temp)/2)
	images.append(zero0)
	maxs.append(0)
	images = np.array(images)

	row,col=np.shape(images[0])
	finalCoord=[]
	for i in range(1,row-1):
		for j in range(1,col-1):
			for k in range(1,len(images)-1):
				img=images[k]
				point=img[i][j]
				if(img[i][j] > min(threshold,maxs[k])):
					grid = images[k-1:k+2,i-1:i+2,j-1:j+2]
					if( np.where(grid>=point)[0].shape[0] == 1):
						finalCoord.append([i,j,k])
	return np.array(finalCoord)


def getSURFFeatures(image, threshold, levels):
	image = cv2.GaussianBlur(image,(3,3),0)
	images = []
	for i in range(1,levels):
		temp=hessian_matrix_det(image, sigma=i)
		temp=removepadd(temp,40)
		cv2.imwrite(str(i)+".jpg",temp*255)
		images.append(temp)

	row,col=np.shape(images[0])
	finalCoord=[]
	for i in range(1,row-1,5):
		for j in range(1,col-1,6):
			maxPt = -1
			coord = [-1,-1,-1]
			for k in range(1,levels):
				img=images[k-1]
				box=img[i-1:i+2,j-1:j+2]
				val = np.amax(box)
				if(val>maxPt):
					maxPt=val
					x,y= np.unravel_index(box.argmax(),box.shape)
					coord = [x-1+i+20,y-1+j+20,k]
			img=np.array(images)[:,i-1:i+2,j-1:j+2]
			if(maxPt>threshold):
				finalCoord.append(coord)
	return np.array(finalCoord)


def displayBlobs(img,coords,name):
	fig, ax = plt.subplots()
	nh,nw= img.shape
	count = 0
	ax.imshow(img, interpolation='nearest',cmap="gray")
	for blob in coords:
		y,x,r = blob[0],blob[1],blob[2]
		c = plt.Circle((x, y), r*1.414, color='red', linewidth=1, fill=False)
		ax.add_patch(c)
	ax.plot()  
	fig.savefig("./querySURFBlob/"+name)
	# plt.show()



def createSURF():
	totData={}
	paths = glob.glob("./images/*.jpg")
	paths.sort()
	for i in paths:
		key=i[9:len(i)-4]
		start=time.time()
		img=cv2.imread(i,0)/255.0
		img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
		coordinates = getSURFFeatures2(img,0.0001,10)
		coordinates= blob._prune_blobs(coordinates,0.3)
		print(i,time.time()-start)
		totData[key]=coordinates.tolist()

	with open('SURFblobs.json', 'w') as fp:
		json.dump(totData, fp)


def createSURFQuery():
	data=getQuery()
	keys=data.keys()
	for i in keys:
		img = data[i]
		start=time.time()
		img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)/255.0
		coordinates = getSURFFeatures2(img,0.001,10)
		coordinates= blob._prune_blobs(coordinates,0.3)
		print(i,time.time()-start)
		displayBlobs(img,coordinates,i+".jpg")
		# totData[key]=coordinates



if __name__ == '__main__':
	# queries = getQuery()

	# createSURF()
	createSURFQuery()

	# temp=cv2.imread("./images/ashmolean_000214.jpg")
	# temp=cv2.resize(temp, (0,0), fx=0.25, fy=0.25)
	# temp=cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)/255.0

	# coordinates = getBlobFeatures(temp,0.01,10)
	# # for i in range(len(coordinates)):
	# # 	coordinates[i]=[coordinates[i][0],coordinates[i][1],coordinates[i][2]]
	# coors= blob._prune_blobs(coordinates,0.3)
	# print(np.shape(coors))
	# des1=getDescriptors(temp,coors)
	# des2=getDescriptors(temp,coordinates)
	# matches =matcher(des1,des2)
	# for i in matches:
	# 	print(i)
	# displayBlobs(temp,coors,"a.jpg")













