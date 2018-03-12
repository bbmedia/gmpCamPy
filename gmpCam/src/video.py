import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/jonas/Desktop/video18-03-10_13-15-59-16.mkv') #Open video file

x = 10
y = 10
xx = 100
yy = 100

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False) #Create the background substractor

kernelOp = np.ones((3,3),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

while(cap.isOpened()):

    ret, frame = cap.read() #read a frame
    
    fgmask = fgbg.apply(frame) #Use the substractor
    
    try:
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        
        #Opening (erode->dilate) para quitar ruido.
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        
        #Closing (dilate -> erode) para juntar regiones blancas.
        mask = cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        
        #frame[x:xx,y:yy] = fgmask;
        
        cv2.imshow('Frame',frame)
        cv2.imshow('fgmask',fgmask)
        cv2.imshow('Background Substraction',mask)
    
    except:
        #if there are no more frames to show...
        print('EOF')
        break
    
    #Abort and exit with 'Q' or ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release() #release video file
cv2.destroyAllWindows() #close all openCV windows
