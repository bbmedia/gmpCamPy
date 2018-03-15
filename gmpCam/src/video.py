##############################################
# G M P C a m Processing
#
# Are we able to count all the T-bar people?
#
# Jonas Walti => jonas.walti@gmail.com
##############################################

import sys
import numpy as np
import cv2
from scipy.spatial import distance
import time
import json

import Rider

targetId = 0

if len(sys.argv) > 1:
    file_obj = open("../config/" + sys.argv[1])
else:
    file_obj  = open('../config/connection.json', 'r')

streams = json.load(file_obj)

cap = cv2.VideoCapture(streams[targetId]["path"])  # Open video file

# processing area of full hd stream
x = streams[targetId]["roi"]["x"]
y = streams[targetId]["roi"]["y"]
w = streams[targetId]["roi"]["w"]
h = streams[targetId]["roi"]["h"]

# minimum area to a a t-bar using human
minArea = 50

# array with all currently processed riders
persons = []

# real t-bar people
realPersons = []

# currently used id
pid = 1

realP = 0

fps = -1
fpsX = 1
fpsCounter = 0

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)  # Create the background substractor

kernelOp = np.ones((3, 3), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

start_time = time.time()

while cap.isOpened():

    # read a frame
    ret, frame = cap.read()

    if ret:
        fgmask = fgbg.apply(frame[y:y + h, x:x + w])  # Use the substractor

        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Opening (erode->dilate)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)

        # Closing (dilate -> erode)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

        # visualize ROI
        frame[y:y + h, x:x + w, 1] = 0

        # find countours
        _, contours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for p in persons:
            p.age_one()

        for cnt in contours0:

            area = cv2.contourArea(cnt)

            if minArea < area < 350:
                # cv2.drawContours(frame[y:y + h, x:x + w], cnt, -1, (0, 255, 0), 3, 8)

                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                xx, yy, ww, hh = cv2.boundingRect(cnt)

                new = True

                for p in persons:

                    if len(p.tracks) > 15 and not p.get_pid() in realPersons:
                        M = distance.cdist(p.tracks, p.tracks, 'euclidean')
                        p.dist = M.max()

                        if p.dist > 80 and p.tracks[0][0] < p.tracks[-1][0]:
                            realPersons.append(p.get_pid())

                    if abs(cx - p.get_x()) <= ww and abs(cy - p.get_y()) <= hh:
                        new = False
                        p.update_position(cx, cy)

                        if len(p.tracks) > 5:
                            cv2.circle(frame[y:y + h, x:x + w], (int(p.get_x()), int(p.get_y())), 5, (p.get_rgb()), -1)
                            img = cv2.rectangle(frame[y:y + h, x:x + w], (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
                            break;

                    if p.timed_out():
                        index = persons.index(p)
                        persons.pop(index)
                        del p

                if new:
                    p = Rider.MyRider(pid, cx, cy, 5)
                    persons.append(p)
                    pid += 1

        frame = frame[300:800, 200:800]

        cv2.putText(frame, "Active processing objects " + str(len(persons)), (20,20), cv2.FONT_HERSHEY_PLAIN, 1, 255)
        cv2.putText(frame, "Total found objects " + str(pid), (20,40), cv2.FONT_HERSHEY_PLAIN, 1, 255)
        cv2.putText(frame, "Detected T-bar riders " + str(len(realPersons)), (20,60), cv2.FONT_HERSHEY_PLAIN, 1, 255)
        cv2.putText(frame, "FPS " + str(fps), (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, 255)

        cv2.imshow('Frame', frame)
        #cv2.imshow('after morph',mask)
        #cv2.imshow('after binerization', imBin)
        #cv2.imshow('after Substraction',fgmask)

        fpsCounter += 1

        if (time.time() - start_time) > fpsX:
            fps = round(fpsCounter / (time.time() - start_time))
            fpsCounter = 0
            start_time = time.time()

        # Abort and exit with 'Q' or ESC
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    else:
        break

# release video file
cap.release()

# close all windows
cv2.destroyAllWindows()
