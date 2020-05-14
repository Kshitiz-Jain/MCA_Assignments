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
from skimage.feature import blob
import json


# def getQuery():
# 	queries = []
# 	paths=glob.glob("./train/query/*query.txt")
# 	paths.sort()
# 	count=0
# 	for i in paths:
# 		q=open(i,"r").readlines()
# 		q=q[0].rstrip("\n").split(" ")
# 		image=cv2.imread("./images/"+q[0][5:]+".jpg")
# 		bbox = [round(float(x)) for x in q[1:]]
# 		cropImage = image[bbox[0]:bbox[0]+bbox[3],bbox[1]:bbox[1]+bbox[2]]
# 		queries.append(cropImage)
# 	return queries



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


def applyfil(img,matfil,ker,r,c):
	a=r-math.floor(ker/2)
	b=c-math.floor(ker/2)
	sum=0.0
	for x in range(ker):
		for y in range(ker):
			sum = sum + (img[a+x][b+y]*matfil[x][y])
	sum=sum/(ker*ker)
	return sum

   
def filter(img,ker,case):
	row,col=img.shape
	pad=math.floor(ker/2)
	newimg = np.zeros((row,col))
	for i in range(row-2*pad):
		for j in range(col-2*pad):
			matfil=lapFil(ker)
			newimg[pad+i][pad+j]=applyfil(img,matfil,ker,pad+i,pad+j)
	return newimg


def getBlobFeatures(image, threshold, levels):
	image = cv2.GaussianBlur(image,(3,3),0)
	images = []
	for i in range(1,levels):
		temp=i**2*ndimage.gaussian_laplace(image, sigma=i)
		# temp=removepadd(temp,40)
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
					coord = [x-1+i,y-1+j,k]
			img=np.array(images)[:,i-1:i+2,j-1:j+2]
			# maxPt=np.amax(img)
			# k,x,y=np.unravel_index(img.argmax(),img.shape)
			# if(coord[2]!=1):
			# print(i,j,maxPt)
			if(maxPt>threshold):
				# finalCoord.append((i+x-1,j+y-1,k))
				finalCoord.append(coord)
	return np.array(finalCoord)


def getBlobFeatures2(image, threshold, levels):
	image = cv2.GaussianBlur(image,(3,3),0)
	images = []
	zero1=np.zeros(np.shape(image))
	zero0=np.zeros(np.shape(image))
	# zero2=removepadd(zero1,40)
	# zero3=removepadd(zero1,40)
	images.append(zero1)
	for i in range(1,levels):
		temp=i**2*ndimage.gaussian_laplace(image, sigma=i)
		# temp=removepadd(temp,40)
		cv2.imwrite(str(i)+".jpg",temp*255)
		images.append(temp)
	images.append(zero0)
	images = np.array(images)

	row,col=np.shape(images[0])
	finalCoord=[]
	for i in range(1,row-1):
		for j in range(1,col-1):
			for k in range(1,len(images)-1):
				img=images[k]
				point=img[i][j]
				if(img[i][j] > threshold):
					grid = images[k-1:k+2,i-1:i+2,j-1:j+2]
					if( np.where(grid>=point)[0].shape[0] == 1):
						finalCoord.append([i,j,k])
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
	fig.savefig("./queryLOGblob/"+name)
	# plt.show()



def getDescriptors(image,coordinates):
	padImage=padding(image,16)
	description = []
	for x,y,z in coordinates:
		block = padImage[x:x+16,y:y+16]
		desBlock = []
		for i in range(4):
			for j in range(4):
				cell = block[i*4:(i+1)*4,j*4:(j+1)*4]
				ll, (lh, hl, hh) = pywt.dwt2(cell, 'haar')
				des=   list(np.ravel(np.sum(lh))) + list(np.ravel(np.sum(hl))) + list(np.ravel(np.absolute(np.sum(lh)))) + list(np.ravel(np.absolute(np.sum(hl)))) 
				desBlock=desBlock+des
		description.append(np.array(desBlock))
	return np.array(description)


def matcher(des1,des2):
	matches=[]
	row1,col1=np.shape(des1)
	row2,col2=np.shape(des2)
	for i in range(row1):
		for j in range(row2):
			dist = des1[i]-des2[j]
			dist = np.sum(dist*dist)
			if(len(matches)<10):
				heapq.heappush(matches, (-1*dist,i,j) )
			else:
				d,obj1,obj2 = heapq.heappop(matches)
				if(-1*d>dist):
					heapq.heappush(matches, (-1*dist,i,j) )
				else:
					heapq.heappush(matches, (d,obj1,obj2))
	return matches


def createBlobDescription():
	totData={}
	paths = glob.glob("./images/*.jpg")
	paths.sort()
	for i in paths:
		key=i[9:len(i)-4]
		start=time.time()
		img=cv2.imread(i,0)/255.0
		img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
		coordinates = getBlobFeatures2(img,0.01,10)
		coordinates= blob._prune_blobs(coordinates,0.3)
		print(i,time.time()-start)
		totData[key]=coordinates.tolist()

	with open('LOGblobs.json', 'w') as fp:
		json.dump(totData, fp)

def createBLOBQuery():
	data=getQuery()
	keys=data.keys()
	for i in keys:
		img = data[i]
		start=time.time()
		img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)/255.0
		coordinates = getBlobFeatures2(img,0.09,10)
		coordinates= blob._prune_blobs(coordinates,0.3)
		print(i,time.time()-start)
		displayBlobs(img,coordinates,i+".jpg")
		exit(0)


if __name__ == '__main__':

	# createBlobDescription()
	createBLOBQuery()
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













