import pytesseract as pt
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from pyzbar import pyzbar

# video:
c=cv2.VideoCapture(0)
p=cv2.CascadeClassifier("cascade/haarcascade_frontalface_alt.xml") # cascede File
while True:
	r, m1=c.read()
	if r==True:
		ret=p.detectMultiScale(m1, minNeighbors=3,minSize=(10,10))
		for x,y,w,h in ret:
			cv2.rectangle(m1,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.imshow("Image 1, m1")
		k=cv2.waitKey(33)
		if k==13:
			break
	else:
		break


# flatImage:
m1 = cv2.imread("people.jpg", 1)

c=cv2.CascadeClassifier("cascade/haarcascade_frontalface_alt.xml")
ret=c.detectMultiScale(m1, minNeighbors=3, minSize=(3,3))

for x,y,w,h in ret:
	cv2.rectangle(m1,(x,y),(x+w,y+h),(0,0,255),2)


cv2.imshow("Image 1", m1)
cv2.waitKey(0)
cv2.destroyAllWindows()

