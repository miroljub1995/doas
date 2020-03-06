from api.camera_reader.reader import CameraReader
from api.dwa.dwa import DWA
from api.yolo import yolo
from api.projective_transformation import transform

import numpy as np
from time import time
import cv2

perspective_size = (5, 8)

def do_for_frame(img):
    img = cv2.resize(img, (416, 416))
    objs, _, labels = obj_det.detect(img)
    lines, obstacles = yolo.lines_and_obstacles(objs, labels)
    perspective_objs = transform.get_perspective_coordinates(objs, perspective_size)
    # create array of obstacles and goal estimation for dwa
    # get angle from dwa
    # turn wheels left/right according to angle from dwa
    cv2.imshow('img', img)
    cv2.waitKey(1)

reader = None
try:
    obj_det = yolo.create()
    dwa = DWA()
    reader = CameraReader()
    # reader = cv2.VideoCapture(0)

    # car initialization here

    while True:
        start = time()
        try: # not to brake program if one frame fail
            ret, img = reader.read()
            print("Read time: {}".format(time() - start))
            do_for_frame(img)
        except Exception as e:
            print("Some error occured: {}".format(e))
        finally:
            end = time()

        print("Time for frame: {}".format(end - start))
finally:
    if reader != None:
        reader.release()