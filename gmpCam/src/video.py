import numpy as np
import cv2

import Rider

cap = cv2.VideoCapture('../media/video18-03-10_13-15-59-16.mkv')  # Open video file

x = 220
y = 500
w = 250
h = 250
minArea = 150

persons = []
pid = 1

realP = 0

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)  # Create the background substractor

kernelOp = np.ones((3, 3), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

while cap.isOpened():

    # read a frame
    ret, frame = cap.read(1)

    if ret:
        fgmask = fgbg.apply(frame[y:y + h, x:x + w])  # Use the substractor

        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Opening (erode->dilate)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)

        # Closing (dilate -> erode)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

        frame[y:y + h, x:x + w, 1] = 0

        _, contours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours0:

            area = cv2.contourArea(cnt)

            if minArea < area < 350:
                # cv2.drawContours(frame[y:y + h, x:x + w], cnt, -1, (0, 255, 0), 3, 8)

                new = True

                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                xx, yy, ww, hh = cv2.boundingRect(cnt)

                for p in persons:
                    if abs(cx - p.get_x()) < w and abs(cy - p.get_y()) < h:
                        new = False
                        p.update_position(cx, cy)

                        if len(p.tracks) > 5:
                            cv2.circle(frame[y:y + h, x:x + w], (cx, cy), 5, (p.get_rgb()), -1)
                            img = cv2.rectangle(frame[y:y + h, x:x + w], (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

                        break;

                    if p.timedOut():
                        index = persons.index(p)
                        persons.pop(index)
                        del p

                if new:
                    p = Rider.MyRider(pid, xx, yy, 4)
                    persons.append(p)
                    pid += 1

        frame = frame[300:800, 200:800]

        cv2.imshow('Frame', frame)
        cv2.imshow('fgmask',fgmask)
        cv2.imshow('Background Substraction',mask)

        # Abort and exit with 'Q' or ESC
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    else:
        break

cap.release()  # release video file
cv2.destroyAllWindows()  # close all openCV windows
