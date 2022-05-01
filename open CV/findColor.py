import cv2
from matplotlib.pyplot import contour
import numpy as np

def empty(a):
    pass

# apply on image

# path = "img/car.jpg"
cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar",640,240)
cv2.createTrackbar("Hue Min","TrackBar",113,179,empty)
cv2.createTrackbar("Hue Max","TrackBar",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBar",48,255,empty)
cv2.createTrackbar("Sat Max","TrackBar",255,255,empty)
cv2.createTrackbar("Val Min","TrackBar",0,255,empty)
cv2.createTrackbar("Val Max","TrackBar",255,255,empty)
cm = cv2.VideoCapture(0) #webcam default 1
cm.set(3,640)
cm.set(4,480)

while True:
    success,img = cm.read()
    # cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBar")
    h_max = cv2.getTrackbarPos("Hue Max","TrackBar")
    s_min = cv2.getTrackbarPos("Sat Min","TrackBar")
    s_max = cv2.getTrackbarPos("Sat Max","TrackBar")
    v_min = cv2.getTrackbarPos("Val Min","TrackBar")
    v_max = cv2.getTrackbarPos("Val Max","TrackBar")
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper) #filtered out image of that color
    imgresult = cv2.bitwise_and(img,img,mask=mask) #check both images and which pixels are same in both img that will return

    # cv2.imshow("Original",img)
    # cv2.imshow("Hsv",imgHSV)
    cv2.imshow("MASK",mask)
    cv2.imshow("Result",imgresult)

    cv2.waitKey(1)
