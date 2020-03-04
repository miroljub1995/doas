from api.camera_reader.reader import CameraReader
from api.yolo import yolo
from api.projective_transformation import transform

import numpy as np
from time import time
import cv2


obj_det = yolo.create()
reader = CameraReader()
# car initialization
perspective_size = (20, 40)

while True:
    start = time()
    try: # not to brake program if one frame fail
        ret, img = reader.read()
        img = cv2.resize(img, (416, 416))
        objs, scores, labels = obj_det.detect(img)
        # perspective_objs = transform.get_perspective_coordinates(objs, perspective_size)
        # create array of obstacles and goal estimation for dwa
        # get angle from dwa
        # turn wheels left/right according to angle from dwa
        end = time()
        print("Time for frame: {}".format(end - start))
    except Exception as e:
        # print("Some error occured: {}".format(err))
        print("Some error occured: {}".format(e))