import numpy as np
import cv2
import argparse
from collections import deque
import statistics
import random



cap=cv2.VideoCapture(0)

# deques for storing center coordinates of object
pts = deque(maxlen=150)
ptsx = deque(maxlen=150)
ptsy = deque(maxlen=150)

# blue color in HSV
Lower_blue = np.array([110,50,50])
Upper_blue = np.array([130,255,255])

#yellow
# Lower_blue = np.array([22, 93, 0])				#BGR
# Upper_blue = np.array([45, 255, 255])

onlyOnePhoto = 0
counter = 0
start = 0
boxFrame = 0
captured = 0									# if captured value = 0, it will write Captured on the frame
hashListStart = [[0]*1000]*1000					# 2d array which tells program when to start recording the trail
hashListStop = [[0]*1000]*1000					# 2d array which tells program when to stop recording the trail

# font = cv2.FONT_HERSHEY_SIMPLEX
# color = (255,0,0)
# org = (50, 50)

dummyRange = list(range(1,1000))






while True:
	ret, img=cap.read()
	blackImg = np.zeros(img.shape)
	img = cv2.flip(img, 1)
	
	cv2.rectangle(img, (150,100), (500,400), color=(255, 0, 0), thickness=3)		#creating a box for capturing the trail only in that frame

	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)					#method is used to convert an image from one color space to another
	kernel=np.ones((5,5),np.uint8)
	mask=cv2.inRange(hsv,Lower_blue,Upper_blue)
	mask = cv2.erode(mask, kernel, iterations=2)			#Erodes away the boundaries of foreground object
	mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)		#for removing noise
	mask = cv2.dilate(mask, kernel, iterations=1)			# Because, erosion removes white noises, but it also shrinks our object. So we dilate it.
	res=cv2.bitwise_and(img,img,mask=mask)					
	cnts,heir=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]			
	center = None
	x = None
	y = None

	
	
 
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		if radius > 5:
			# cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(img, center, 5, (0, 0, 255), -1)
		
	pts.appendleft(center)
	
	flag = 0
	if captured:
		cv2.putText(img, 'Captured', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

	if(x is not None and y is not None):
		flag = 1
		ptsx.appendleft(int(x))
		ptsy.appendleft(int(y))
	else:
		ptsx.appendleft(int(-1))
		ptsy.appendleft(int(-1))

	flag = False
	
	for i in range (1,len(pts)):
			
		if pts[i-1]is None or pts[i] is None:
			continue
		if ptsx[i-1] == -1 or ptsx[i] == -1 or ptsy[i-1] == -1 or ptsy[i] == -1:
			continue 

		isInside = 150 < pts[i][0] < 500 and 100 < pts[i][1] < 400

		if(isInside):
			if(start == 0):
				hashListStart[ptsx[i]][ptsy[i]] += 1

			# print(ptsx[i], ptsy[i])

			if(hashListStart[ptsx[i]][ptsy[i]] > 150):
				start = 1
				hashListStart = [[0]*1000]*1000
				# hashListStop = [[0]*1000]*1000
			
			if(start == 1):
				cv2.rectangle(img, (150,100), (500,400), color=(0, 128, 0), thickness=3)	# for changing rectangle box color to green
				boxFrame = 1
				thick = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
				
				cv2.line(img, pts[i-1] , pts[i] ,(0,0,225),thick)
				if i>10:
					cv2.line(blackImg,pts[i-10] , pts[i-9] ,(225,225,225),thick*2)
				# cv2.line(blackImg,pts[i-1] , pts[i] ,(225,225,225),thick*2)
				# print(i, ptsx[i], ptsy[i])

				if(onlyOnePhoto == 0) :
					
					hashListStop[ptsx[i]][ptsy[i]] += 1

					if(hashListStop[ptsx[i]][ptsy[i]] > 800):
						
						crop_img = blackImg[100:400, 150:500]

						fileName = str(random.choice(dummyRange))
						cv2.imwrite("Images/sample"+ fileName +".jpg", crop_img)
						print("Captured" + fileName)
						captured = 1
						
						
						onlyOnePhoto = 1
						boxFrame = 1
						start = 0
						hashListStop = [[0]*1000]*1000
						flag = True

	if flag:
		break
			
	
	cv2.imshow("Frame", img)
	cv2.imshow("Frame2", blackImg)
	# cv2.imshow("mask",mask)
	# cv2.imshow("res",res)
	
	
	
	k=cv2.waitKey(30) & 0xFF
	if k==32:
		break


# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
