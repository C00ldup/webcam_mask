import numpy as np
import cv2 as cv

flags = [i for i in dir(cv) if i.startswith('COLOR_')]
#print( flags )

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
    
faceCascade = cv.CascadeClassifier("opencv\/data\/haarcascades\/haarcascade_frontalface_default.xml")
eyeCascade = cv.CascadeClassifier('opencv\/data\/haarcascades\/haarcascade_eye.xml')
    
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    #if len(faces)>0:
    #    cv.rectangle(frame, (faces[0][0], faces[0][1]), (faces[0][0]+faces[0][2], faces[0][1]+faces[0][3]), (255, 0, 0), 2)
    for (x,y, w, h) in faces:
        cv.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness = 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=10)
        for (ex,ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
     # define range of blue color in HSV
    lower = findHSVColor(141, 85, 36)
    upper = findHSVColor(255, 219, 172)
    mask = cv.inRange(hsv, lower, upper)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    
    # mirroring image
    frame = cv.flip(frame,1)
    # Display the resulting frame
    cv.imshow('face_frame', frame)
    '''
    cv.imshow('grigio', gray)
    cv.imshow('original', frame)
    cv.imshow('hsv', hsv)
    cv.imshow('mask', mask)
    cv.imshow('result', res)
    cv.imshow('colors',img)
    '''
    
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