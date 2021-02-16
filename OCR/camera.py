import cv2
from model import FacialExpressionModel
import numpy as np

from PIL import Image
import easyocr
import time

ocr_engine = easyocr.Reader(['en'])

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        # self.video = cv2.VideoCapture("/home/sanket/Downloads/Facial expression recognition/videos/facial_exp.mkv")
        self.video = cv2.VideoCapture(0) # for camera capture

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        time.sleep(5)
        img = Image.fromarray(fr, 'RGB')
        img.save('my.png')
        print(type(img))
        try:
            res = ocr_engine.readtext('my.png')
            print(res)
        except Exception as e:
            print('error:', e)
        # for b, t, c in res:
        #     print(t)
        #     print(c)
        #     cv2.putText(fr, res, (b[0], b[1]), font, 1, (255, 255, 0), 2)
        # img.save('my.png')
        # gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        # faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        # for (x, y, w, h) in faces:
        #     fc = gray_fr[y:y+h, x:x+w]

        #     roi = cv2.resize(fc, (48, 48))
        #     pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        # cv2.putText(fr, res, (boxes[0], boxes[1]), font, 1, (255, 255, 0), 2)
            # cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
