from api.camera_reader.reader import CameraReader
from api.dwa.dwa import DWA
from api.yolo import yolo
from api.projective_transformation import transform

import numpy as np
from time import time
import cv2
import math
import traceback

perspective_size = (5, 8)




def draw(img, pixels):
    for pix in pixels:
        x, y = pix
        cv2.circle(img, (x, y), 1, 255, 1)




def do_for_frame(img):
    img = cv2.resize(img, (416, 416))
    objs, _, labels = obj_det.detect(img)
    lines, obstacles = yolo.lines_and_obstacles(objs, labels)
    print('*************************')
    print("Lines: {}".format(lines))
    #print("Obstacles: {}".format(obstacles))

    if len(lines) == 2:    
        perspective_lines = transform.get_perspective_coordinates(lines, perspective_size)
        goal = yolo.calculate_goal(perspective_lines)
        perspective_obstacles = transform.get_perspective_coordinates(obstacles, perspective_size)
        print("Obs: {}".format(perspective_obstacles))

        test_image = np.zeros((10, 10, 1), dtype=np.uint8)
        draw(test_image, perspective_obstacles)
        cv2.imshow('persp', test_image)

        curr = (2, 10)
        curr_angle = -math.pi / 2.0
        angle = dwa.calculate_angle(goal, perspective_obstacles, curr)
    
        print("Goal: {}".format(goal))
        print("Angle: {} (from [-1, 1])".format(angle))
        
        
    #create array of obstacles and goal estimation for dwa
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
            #print("Read time: {}".format(time() - start))
            do_for_frame(img)
        except Exception as e:
            print("Some error occured: {}".format(e))
            traceback.print_exc()
            raise e
        finally:
            end = time()

        print("Time for frame: {}".format(end - start))
finally:
    if reader != None:
        reader.release()