##############################################
# G M P C a m Processing
#
# Are we able to count all the T-bar people?
#
# Jonas Walti => jonas.walti@gmail.com
##############################################

# import the necessary packages
from flask import Response
from flask import Flask
from flask import abort
import threading
import cv2
import json
import sys
import time

from src.tbardetector import TBarDetector, ImageSlices

targetId = 1

if len(sys.argv) > 1:
    print(sys.argv[1])
    file_obj = open("config/" + sys.argv[1])
else:
    file_obj = open('config/connection.json', 'r')

settingsF = open('config/general.json')
settings = json.load(settingsF)

pwdsF = open('config/pwds.json')
pwds = json.load(pwdsF)

streams = json.load(file_obj)

detector = TBarDetector()

currentStreamId = ''
cap = None

# processing area of full hd stream
x = streams[targetId]["roi"]["x"]
y = streams[targetId]["roi"]["y"]
w = streams[targetId]["roi"]["w"]
h = streams[targetId]["roi"]["h"]

try:
    liveX = streams[targetId]["liveView"]["x"]
    liveY = streams[targetId]["liveView"]["y"]
    liveW = streams[targetId]["liveView"]["w"]
    liveH = streams[targetId]["liveView"]["h"]
    liveViewSet = True

    detector.set_show_image(True, ImageSlices(liveX, liveY, liveW, liveH))
except KeyError:
    liveViewSet = False

outputframe = None
latest_update = time.time()

lock = threading.Lock()

app = Flask(__name__)


def process_stream():
    global cap, currentStreamId, outputframe, lock, latest_update

    while True:

        if cap is None:
            continue

        if not cap.isOpened():
            continue

        # read a frame
        if time.time() - latest_update > 1/24:

            ret, frame = cap.read()
            latest_update = time.time()

            if ret:
                with lock:
                    outputframe = detector.process_image(frame, x, y, w, h)
            else:
                print("error opening frame! ")
                cap.release()


def generate():

    # grab global references to the output frame and lock variables
    global outputframe, lock

    # loop over frames from the output stream
    while True:

        # wait until the lock is acquired
        with lock:

            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputframe is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputframe)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed/<stream_id>")
def video_feed(stream_id):

    global cap, currentStreamId

    # return the response generated along with the specific media
    # type (mime type)

    if currentStreamId != stream_id:

        if cap is not None:
            cap.release()

        stream = [stream for stream in streams if stream['id'] == stream_id]
        if len(stream) == 0:
            abort(404)

        if cap is not None:
            cap.release()

        cap = cv2.VideoCapture(stream[0]['path'])  # Open video file
        currentStreamId = stream[0]['id']

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':

    # start a thread that will perform motion detection
    t = threading.Thread(target=process_stream)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host="0.0.0.0", port="8000", debug=True,
            threaded=True, use_reloader=False)
