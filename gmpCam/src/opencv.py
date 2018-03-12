import cv2

img = cv2.imread('imageCurrent00014.jpg',0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

surf = cv2.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img,None)

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
