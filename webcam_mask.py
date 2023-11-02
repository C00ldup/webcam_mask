import numpy as np
import cv2 as cv

flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print( flags )

def findHSVColor(r,g,b):
    hsv_green = cv.cvtColor(np.uint8([[[r,g,b]]]),cv.COLOR_BGR2HSV)
    return hsv_green[0][0]

def nothing(x):
    pass
# Create a black colors, a window
img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('colors')
# create trackbars for color change
cv.createTrackbar('R','colors',0,255,nothing)
cv.createTrackbar('G','colors',0,255,nothing)
cv.createTrackbar('B','colors',0,255,nothing)
# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'colors',0,1,nothing)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
     # define range of blue color in HSV
    lower = findHSVColor(141, 85, 36)
    upper = findHSVColor(255, 219, 172)
    
    lower = np.array([110,50,50])
    upper = np.array([130,255,255])
    
    mask = cv.inRange(hsv, lower, upper)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    
    # Display the resulting frame
    cv.imshow('original', frame)
    cv.imshow('hsv', hsv)
    cv.imshow('mask', mask)
    cv.imshow('result', res)
    cv.imshow('colors',img)
    
    # get current positions of four trackbars
    r = cv.getTrackbarPos('R','colors')
    g = cv.getTrackbarPos('G','colors')
    b = cv.getTrackbarPos('B','colors')
    s = cv.getTrackbarPos(switch,'colors')
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()