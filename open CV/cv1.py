import cv2
from matplotlib.pyplot import contour
import numpy as np
kernel = np.ones((5,5),np.uint8) #make 5x5 once array and convert in 0-255 bit values
# img = cv2.imread("img/dog.jpg")
# cv2.imshow("Dog view",img) # show image
# cv2.waitKey(0) #load image (time in milisec.) 0 means infinite
"""
Video 

vd = cv2.VideoCapture("img/dogvd.mp4") #load video
while True:
    success,img = vd.read()
    cv2.imshow("dog video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""

"""
Webcam

cm = cv2.VideoCapture(0) #webcam default 1
cm.set(3,640)
cm.set(4,480)
while True:
    success,img = cm.read()
    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""

"""
play with images :)
functions
"""
img = cv2.imread("img/dog.jpg")
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurimg = cv2.GaussianBlur(img,(7,7),0) #blur image(img,array,sigma:X)
cannyimg = cv2.Canny(img,100,150)#edge of image (img,edge visibility)
# dilateimg = cv2.dilate(cannyimg,kernel,iterations=1) # edge width
# erodeimg = cv2.erode(dilateimg,kernel,iterations=1) # reduce edge width(opp. dilateimg) 

cv2.imshow("original",img)
cv2.imshow("blur",blurimg)
cv2.imshow("Gray",grayimg)
cv2.imshow("Edge",cannyimg)
# cv2.imshow("Edgewidth",dilateimg)
# cv2.imshow("erodewidth",erodeimg)

cv2.waitKey(0)


"""
Crop and Resize
img = cv2.imread("img/dog.jpg")
resizeimg = cv2.resize(img,(150,200)) #resize image (width,height)
croppedimg = img[0:100,100:150]#crop image(top:depth,from left(start:100):right(end:150))

cv2.imshow("Original image",img)
cv2.imshow("Resize image",resizeimg)
cv2.imshow("Cropped image",croppedimg)

cv2.waitKey(0)
"""
"""
Play with shapes


img = np.zeros((550,550,3),np.uint8)
# img[:] = 0,0,0 #color whole image

cv2.line(img,(0,0),(300,300),(0,255,0),2)
cv2.rectangle(img,(0,0),(200,200),(230,55,40),3)
cv2.circle(img,(300,100),40,(255,255,0),2)
cv2.putText(img,("Shyam Patel"),(200,330),cv2.FONT_ITALIC,1,(0,150,250),2)
showimg = cv2.imshow("black_img",img)
cv2.waitKey(0)

"""
"""
Join Images


img = cv2.imread("img/bean.jpg")

imghz = np.hstack((img,img))
imgvr = np.vstack((img,img))

cv2.imshow("horizontal",imghz)
cv2.imshow("vertical",imgvr)

check stackImages() function for print all images in stack

cv2.waitKey(0)
"""

"""Image object detection

def empty(a):
    pass

# apply on image

path = "img/car.jpg"
cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar",640,240)
cv2.createTrackbar("Hue Min","TrackBar",12,179,empty)
cv2.createTrackbar("Hue Max","TrackBar",36,179,empty)
cv2.createTrackbar("Sat Min","TrackBar",70,255,empty)
cv2.createTrackbar("Sat Max","TrackBar",255,255,empty)
cv2.createTrackbar("Val Min","TrackBar",167,255,empty)
cv2.createTrackbar("Val Max","TrackBar",255,255,empty)

while True:
    img = cv2.imread(path)
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

    cv2.imshow("Original",img)
    # cv2.imshow("Hsv",imgHSV)
    cv2.imshow("MASK",mask)
    cv2.imshow("Result",imgresult)

    cv2.waitKey(1)


"""

"""Shape Detection


def getcontours(img):#Get edge value of shape
    
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#find shape
    for cnt in contours:
        area = cv2.contourArea(cnt) #find area
        if area>500: #shape area >500 then run
            cv2.drawContours(imgcontour,cnt,-1,(255,0,0),2)#draw shape border
            peri = cv2.arcLength(cnt,True) #find arc points
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)#round value of arc points
            objcor = len(approx) #total corner points 
            x,y,w,h = cv2.boundingRect(approx)#calculate box around whole shape area so we can find center point easily
            if objcor == 3:
                ObjectType = "Tri"
            elif objcor == 4: #there are 2 possiblity for 4 corner sqr or rect
                aspRatio = w/float(h)#for sqr width and height is same so divide both and ans. is nearly 1
                if aspRatio>0.95 and aspRatio<1.05:#apply this condition for square
                    ObjectType = "SQR"
                else:
                    ObjectType = "RECT" 
            elif objcor > 5:#circle has many corners most of >5
                ObjectType = "Circle"
            else:
                ObjectType = "None"
            cv2.rectangle(imgcontour,(x,y),(x+w,y+h),(0,255,0),2) #draw box
            cv2.putText(imgcontour,ObjectType,(x+(w//2)-20,y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2)#put text at the middle on shape accourding to its type

img = cv2.imread("img/shapes.jpg")  #read img
imgcontour = img.copy() #make copy of img

imgBW = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to grayscale
imgblur = cv2.GaussianBlur(imgBW,(7,7),1) #blue(not imp)
imgedge = cv2.Canny(img,100,100) #find edge
getcontours(imgedge) #call function pass img edge
imgblank = np.zeros_like(img) #create blank img

cv2.imshow("original",img)
# cv2.imshow("B/W",imgBW)
# cv2.imshow("Blur",imgblur)
# cv2.imshow("Edge",imgedge)
# cv2.imshow("Blank",imgblank)
cv2.imshow("Predict",imgcontour)

cv2.waitKey(0)

"""
"""Face Detection

img = cv2.imread("img/face.jpg") #read image
face_cascade = cv2.CascadeClassifier('img/haarcascade_frontalface_default.xml') #apply face detection library
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray image
faces = face_cascade.detectMultiScale(imggray, 1.1, 4) #scale factor > 1
# Draw the rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Detect",img)
cv2.waitKey(0)

"""

"""Live Face Detection

face_cascade = cv2.CascadeClassifier('img/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the VideoCapture object
cap.release()

"""
#PROJECT 1 VIRTUAL PAINT

# cm = cv2.VideoCapture(0) #webcam default 0
# cm.set(3,640)#(id,frame width)
# cm.set(4,480)#(id,frame height)
# cm.set(10,150)
# #value from object detection Trackbar
# myColors = [[113,48,0,179,255,255]]#pink
#             # [104,126,0,115,212,162], #blue color [min,min,min,max,max,max]
#             # [0,57,64,20,255,255],#skin
#             # [0,99,97,61,139,255],#red
#             # [21,75,0,29,255,255],#ye color
#             # [35,0,0,179,24,85]#black
#             # ,[20,0,9,104,23,97],#white
#             # [57,76,0,100,255,255]]#green
#             # ] #skin color(hand,face)

# myColorsValue = [[127,0,255]]#pink   BGR format
#                 # [255,0,0], #blue
#                 # [153,153,255],#skin
#                 # [0,0,255],#red
#                 # [0,255,255],#yellow
#                 # [0,0,0],#black
#                 # [255,255,255],#white
#                 # [0,255,0]]#green

# myPoints = [] #[x,y,colorid] colorid = mycolorvalue if 0 then 255,0,0

# def findColor(img,myColors,myColorsValue):
#     imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     c=0 #for getting color values
#     newPoints = []
#     for color in myColors:
#         lower = np.array(color[0:3]) #lower = np.array([h_min,s_min,v_min])
#         upper = np.array(color[3:6])
#         mask = cv2.inRange(imgHSV,lower,upper)
#         x,y = getcontours(mask) #store center value
#         cv2.circle(imgResult,(x,y),10,myColorsValue[c],cv2.FILLED) #display circle at returned value point
#         if x!=0 and y!=0:
#             newPoints.append([x,y,c])
#         c+=1
#         # cv2.imshow(str(color[0]),mask) #3 output window is generated,in each iteration we changed show window name if we put it same then only last values window we get
#     return newPoints

# def getcontours(img):#Get edge value of shape
#     contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#find shape
#     x,y,w,h = 0,0,0,0 # if area<500 then return 0
#     for cnt in contours:
#         area = cv2.contourArea(cnt) #find area
#         if area>500: #shape area >500 then run
#             cv2.drawContours(imgResult,cnt,-1,(255,0,0),2)#draw shape border
#             peri = cv2.arcLength(cnt,True) #find arc points
#             approx = cv2.approxPolyDP(cnt,0.02*peri,True)#round value of arc points
#             x,y,w,h = cv2.boundingRect(approx)
#     return x+w//2,y # return center of top edge

# def drawOnCanvas(myPoints,myColorValues):
#     for point in myPoints:
#         cv2.circle(imgResult,(point[0],point[1]),10,myColorsValue[point[2]],cv2.FILLED)

# while True:
#     success,img = cm.read()
#     imgResult = img.copy()
#     newPoints = findColor(img,myColors,myColorsValue)
#     if len(newPoints) != 0:
#         for npt in newPoints: #we get newPoint as list so we can't put it inside list
#             myPoints.append(npt) #append each point
#     if len(myPoints) != 0:
#         drawOnCanvas(myPoints,myColorsValue)
#     cv2.imshow("Result",imgResult)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break