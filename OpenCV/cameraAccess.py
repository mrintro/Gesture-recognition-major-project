import numpy as np
import cv2
import argparse
from collections import deque
import statistics
import random
from predict import *

cnn_model = load_cnn_model()


cap=cv2.VideoCapture(0)

# deques for storing center coordinates of object
centroidQueue = deque(maxlen=150)
xQueue = deque(maxlen=150)
yQueue = deque(maxlen=150)

# blue color in HSV
Lower_blue = np.array([110,50,50])
Upper_blue = np.array([130,255,255])

#yellow
# Lower_blue = np.array([22, 93, 0])				
# Upper_blue = np.array([45, 255, 255])

onlyOnePhoto = 0
counter = 0
start = 0
boxFrame = 0
captured = 0									# if captured value = 0, it will write Captured on the frame
hashListStart = [[0]*1000]*1000					# 2d array which tells program when to start recording the trail
hashListStop = [[0]*1000]*1000					# 2d array which tells program when to stop recording the trail
prediction = 0
ClosingTimer = 50

while True:
	ret, img=cap.read()
	blackImg = np.zeros(img.shape)
	img = cv2.flip(img, 1)
	
	cv2.rectangle(img, (150,100), (500,400), color=(255, 0, 0), thickness=3)		#creating a box for capturing the trail only in that frame

	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)					#method is used to convert an image from one color space to another
	kernel=np.ones((5,5),np.uint8)
	originalMask=cv2.inRange(hsv,Lower_blue,Upper_blue)
	originalMask = cv2.erode(originalMask, kernel, iterations=2)			#Erodes away the boundaries of foreground object
	originalMask=cv2.morphologyEx(originalMask,cv2.MORPH_OPEN,kernel)		#for removing noise
	originalMask = cv2.dilate(originalMask, kernel, iterations=1)			# Because, erosion removes white noises, but it also shrinks our object. So we dilate it.
	# res=cv2.bitwise_and(img,img,mask=originalMask)					
	cnts,heir=cv2.findContours(originalMask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]			
	# coordinates
	center = None
	x = None
	y = None

	
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		if radius > 5:
			cv2.circle(img, center, 5, (0, 0, 255), -1)
	
	centroidQueue.appendleft(center)								# appending centroids to the queue
	
	flag = 0
	
	# pushing x and y coordinates only if they exists
	if(x is not None and y is not None):
		flag = 1
		xQueue.appendleft(int(x))
		yQueue.appendleft(int(y))
	else:
		xQueue.appendleft(int(-1))
		yQueue.appendleft(int(-1))

	flag = False
	
	for i in range (1,len(centroidQueue)):
			
		if centroidQueue[i-1]is None or centroidQueue[i] is None:
			continue
		if xQueue[i-1] == -1 or xQueue[i] == -1 or yQueue[i-1] == -1 or yQueue[i] == -1:
			continue 

		isInside = 150 < centroidQueue[i][0] < 500 and 100 < centroidQueue[i][1] < 400

		if(isInside):
			if(start == 0):
				hashListStart[xQueue[i]][yQueue[i]] += 1

			if(hashListStart[xQueue[i]][yQueue[i]] > 150):
				start = 1
				hashListStart = [[0]*1000]*1000
			
			if(start == 1):
				cv2.rectangle(img, (150,100), (500,400), color=(0, 128, 0), thickness=3)	# for changing rectangle box color to green
				boxFrame = 1
				thick = int(np.sqrt(len(centroidQueue) / float(i + 1)) * 2.5)
				
				cv2.line(img, centroidQueue[i-1] , centroidQueue[i] ,(0,0,225),thick)
				if i>10:
					cv2.line(blackImg,centroidQueue[i-10] , centroidQueue[i-9] ,(225,225,225),thick)
				# cv2.line(blackImg,centroidQueue[i-1] , centroidQueue[i] ,(225,225,225),thick*2)

				if(onlyOnePhoto == 0) :
					
					hashListStop[xQueue[i]][yQueue[i]] += 1

					if(hashListStop[xQueue[i]][yQueue[i]] > 1000):
						crop_img = blackImg[100:400, 150:500]
						cv2.imwrite("Images/sample.jpg", crop_img)
						prediction, confidence = predict(cnn_model, cv2.imread('Images/sample.jpg'))
						print("Captured")
						print("predicted Number", prediction)
						print("Confidence", confidence)
						
						captured = 1
						onlyOnePhoto = 1
						boxFrame = 1
						start = 0
						hashListStop = [[0]*1000]*1000
						
	if captured:
		cv2.putText(img, "Predicted : " + str(prediction) , (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
		cv2.putText(img, "Confidence : " + str(int(confidence*100)) , (300,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
		ClosingTimer = ClosingTimer-1
		if(ClosingTimer < 0):
			exit()
			
	
	cv2.imshow("Frame", img)
	# cv2.imshow("Frame2", blackImg)
	# cv2.imshow("originalMask",originalMask)
	# cv2.imshow("res",res)
	
	
	
	k=cv2.waitKey(30) & 0xFF
	if k==32:
		break


# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
