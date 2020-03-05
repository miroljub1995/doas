from api.camera_reader.reader import CameraReader, CameraReader1
from api.dwa.dwa import DWA
from api.yolo import yolo
from api.projective_transformation import transform

import numpy as np
from time import time
import cv2


obj_det = yolo.create()
dwa = DWA()
reader = None
try:
    # reader = CameraReader()
    reader = CameraReader1()
    # reader = cv2.VideoCapture(0)

    # car initialization here
    perspective_size = (20, 40)
    while True:
        start = time()

        try: # not to brake program if one frame fail
            ret, img = reader.read()
            print("Read time: {}".format(time() - start))
            img = cv2.resize(img, (416, 416))
            objs, scores, labels = obj_det.detect(img)
            # perspective_objs = transform.get_perspective_coordinates(objs, perspective_size)
            # create array of obstacles and goal estimation for dwa
            # get angle from dwa
            # turn wheels left/right according to angle from dwa
            cv2.imshow('img', img)
            cv2.waitKey(1)
            end = time()
        except Exception as e:
            print("Some error occured: {}".format(e))

        print("Time for frame: {}".format(end - start))
finally:
    if reader != None:
        reader.release()