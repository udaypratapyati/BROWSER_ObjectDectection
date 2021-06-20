import numpy as np
import cv2

from flask import Response
from flask import Flask
from flask import render_template
import threading
import time

PROTOTXT = "model/MobileNetSSD_deploy.prototxt"
MODEL = "model/MobileNetSSD_deploy.caffemodel"
INP_VIDEO_PATH = 'cars.mp4'
OUT_VIDEO_PATH = 'cars_detection.mp4'
GPU_SUPPORT = 0
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",  "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

lock = threading.Lock()
outputFrame = None

# initialize a flask object
app = Flask(__name__)

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
cap = cv2.VideoCapture(0)

time.sleep(2.0)

def detect():

    global net, cap, outputFrame, lock

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],confidence*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                with lock:
                    outputFrame = frame.copy()

    cap.release()
 
def generate():
    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

if __name__ == '__main__':
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=True, use_reloader=False)
