import cv2
import numpy as np
from scipy.spatial import distance
from src.Rider import MyRider
import time


class ImageRect:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class ImageSlices:

    def __init__(self, x, y, w, h):
        self.width = slice(x, x + w)
        self.height = slice(y, y + h)


class TBarDetector:

    def __init__(self):

        self.fpsCounter = 0
        # open cv background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)  # Create the background subtractor

        # convolution kernels
        self.kernelOp = np.ones((3, 3), np.uint8)
        self.kernelCl = np.ones((11, 11), np.uint8)

        self.fps = -1
        self.fpsCounter = 0

        self.start_time = time.time()

        # minimum area to a t-bar using human
        self.minArea = 50
        self.maxArea = 1000

        self.pid = 1

        # array with all currently processed riders
        self.persons = []

        # real t-bar people
        self.realPersons = []

        self.showImage = True
        self.showImageSize = ImageSlices(10, 10, 100, 100)

    def set_show_image(self, on, size):

        self.showImage = on
        self.showImageSize = size

    def process_image(self, frame, x, y, w, h):

            fgmask = self.fgbg.apply(frame[y:y + h, x:x + w])  # Use the substractor

            ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

            # Opening (erode->dilate)
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, self.kernelOp)

            # Closing (dilate -> erode)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernelCl)

            # visualize ROI
            frame[y:y + h, x:x + w, 1] = 0

            # find contours
            contours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for p in self.persons:
                p.age_one()

                if p.timed_out():
                    index = self.persons.index(p)
                    self.persons.pop(index)
                    del p

            for cnt in contours0:

                area = cv2.contourArea(cnt)

                if self.minArea < area < self.maxArea:
                    # cv2.drawContours(frame[y:y + h, x:x + w], cnt, -1, (0, 255, 0), 3, 8)

                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    xx, yy, ww, hh = cv2.boundingRect(cnt)

                    new = True

                    for p in self.persons:

                        if len(p.tracks) > 15 and not p.get_pid() in self.realPersons:
                            M = distance.cdist(p.tracks, p.tracks, 'euclidean')
                            p.dist = M.max()

                            # simply check the minimal travelled dist
                            if p.dist > 80 and p.tracks[0][0] < p.tracks[-1][0]:
                                self.realPersons.append(p.get_pid())
                                #TODO: toUploadP += 1

                        if abs(cx - p.get_x()) <= ww and abs(cy - p.get_y()) <= hh:
                            new = False
                            p.update_position(cx, cy)

                            if len(p.tracks) > 5:
                                cv2.circle(frame[y:y + h, x:x + w], (int(p.get_x()), int(p.get_y())), 5, (p.get_rgb()),
                                           -1)
                                img = cv2.rectangle(frame[y:y + h, x:x + w], (xx, yy), (xx + ww, yy + hh), (0, 255, 0),
                                                    2)
                                break;

                    if new:
                        p = MyRider(self.pid, cx, cy, 5)
                        self.persons.append(p)
                        self.pid += 1

            if self.showImage:
                frame = frame[self.showImageSize.height, self.showImageSize.width]

                cv2.putText(frame, "Active processing objects " + str(len(self.persons)), (20, 20), cv2.FONT_HERSHEY_PLAIN,
                            1, 255)
                cv2.putText(frame, "Total found objects " + str(self.pid), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, 255)
                cv2.putText(frame, "Detected T-bar riders " + str(len(self.realPersons)), (20, 60), cv2.FONT_HERSHEY_PLAIN,
                            1, 255)
                cv2.putText(frame, "FPS " + str(self.fps), (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, 255)

                #cv2.imshow('Frame', frame)
                #cv2.imshow('after morph', mask)
                #cv2.imshow('after binerization', imBin)
                #cv2.imshow('after Substraction', fgmask)

            self.fpsCounter += 1

            if (time.time() - self.start_time) > 5:
                self.fps = round(self.fpsCounter / (time.time() - self.start_time))
                self.fpsCounter = 0
                self.start_time = time.time()

            return frame
