##############################################
# G M P C a m Processing
#
# Are we able to count all the T-bar people?
#
# Jonas Walti => jonas.walti@gmail.com
##############################################

import sys
import cv2
import time
import datetime
import json

from src.tbardetector import TBarDetector
from src.tbardetector import ImageSlices
from src.DataSaverSQL import DataSaverSQL

targetId = 1

if len(sys.argv) > 1:
    print(sys.argv[1])
    file_obj = open("../config/" + sys.argv[1])
else:
    file_obj = open('../config/connection.json', 'r')

settingsF = open('../config/general.json')
settings = json.load(settingsF)

pwdsF = open('../config/pwds.json')
pwds = json.load(pwdsF)

streams = json.load(file_obj)

detector = TBarDetector()

try:
    cap = cv2.VideoCapture(streams[targetId]["path"])  # Open video file
except IndexError:
    print("object index wrong! Stop everything!")
    exit()

# processing area of full hd stream
x = streams[targetId]["roi"]["x"]
y = streams[targetId]["roi"]["y"]
w = streams[targetId]["roi"]["w"]
h = streams[targetId]["roi"]["h"]

try:
    stopAt = datetime.time(settings["stopH"],
                           settings["stopM"],
                           settings["stopS"])
    autoStop = True
except KeyError:
    autoStop = False

try:
    isDemo = settings["isDemo"]
except KeyError:
    isDemo = False

# live view
liveViewSet = True
try:
    liveX = streams[targetId]["liveView"]["x"]
    liveY = streams[targetId]["liveView"]["y"]
    liveW = streams[targetId]["liveView"]["w"]
    liveH = streams[targetId]["liveView"]["h"]
    liveViewSet = True

    detector.set_show_image(True, ImageSlices(liveX, liveY, liveW, liveH))


except KeyError:
    liveViewSet = False

# time measurements
startMeasureForUpload = time.time()

# stats upload
toUploadP = 0
uploadIntervalS = 20

# Connect to the database
sqlConn = DataSaverSQL(settings["sqlConnection"], settings["sqlUser"], pwds["sql"], settings["sqlDbName"])
sqlConn.connect()

while cap.isOpened():

    # read a frame
    ret, frame = cap.read()

    if ret:

        detector.process_image(frame, x, y, w, h)

        if time.time() - startMeasureForUpload > uploadIntervalS and toUploadP > 0:
            print("upload " + str(toUploadP))
            sqlConn.count_up(toUploadP, isDemo)
            toUploadP = 0
            startMeasureForUpload = time.time()

            # Abort and exit with 'Q' or ESC
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        if autoStop and not isDemo and stopAt < datetime.datetime.now().time():
            break

    else:
        print("error opening frame! ")
        cap.release()
        time.sleep(10)
        try:
            cap = cv2.VideoCapture(streams[targetId]["path"])
        except:
            print("cannot reopen stream. close app.")
            break

d = datetime.datetime.now()
print("stopping script " + d.strftime("%H:%M:%S %d.%m.%y"))

# release video file
cap.release()

# close all windows
cv2.destroyAllWindows()

sqlConn.connection.close()
