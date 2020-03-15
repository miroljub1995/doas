from time import time
import math
import traceback
import os

import cv2
import numpy as np

from api.camera_reader.reader import CameraReader
from api.dwa.dwa import DWA
from api.yolo import yolo
from api.projective_transformation import transform

perspective_size = (500, 800)

def draw_pixels(img, pixels):
    for pix in pixels:
        x, y = pix
        cv2.circle(img, (x, y), 1, 255, 1)


def do_for_frame(img, obj_detector, dwa):
    img = cv2.resize(img, (416, 416))
    objs, _, labels = obj_detector.detect(img)
    obj_detector.draw(img, [objs, _, labels])
    print(objs)
    lines, obstacles = yolo.lines_and_obstacles(objs, labels)
    print('*************************')
    print("Lines: {}".format(lines))
    #print("Obstacles: {}".format(obstacles))

    if len(lines) == 2:    
        perspective_lines = transform.get_perspective_coordinates(lines, perspective_size)
        print("Real obs: {}".format(obstacles))
        goal = yolo.calculate_goal(perspective_lines)
        perspective_obstacles = transform.get_perspective_coordinates(obstacles, perspective_size)
        perspective_lines = transform.get_perspective_coordinates(lines, perspective_size)
        wrapped = transform.get_wrapped_img(img, perspective_size)
        cv2.imshow('wrapped', wrapped)
        cv2.waitKey(1)

        print("Perspective obs: {}".format(perspective_obstacles))
        obs_pixels = transform.rects_to_pixels(perspective_obstacles, perspective_size)
        print("Obs: {}".format(obs_pixels))

        test_image = np.zeros((perspective_size[1], perspective_size[0], 1), dtype=np.uint8)
        for line in perspective_lines:
            x1, y1, x2, y2 = line
            cv2.rectangle(test_image, (x1, y1), (x2, y2), 255, 1)
            px1, py1, px2, py2 = int((x1 + x2) // 2), y1, int((x1 + x2) // 2), y2
            cv2.line(test_image, (px1, py1), (px2, py2), 255, 1)
        draw_pixels(test_image, obs_pixels)
        cv2.imshow('persp', test_image)
        cv2.waitKey(1)

        curr = (2, 10)
        curr_angle = -math.pi / 2.0
        angle = dwa.calculate_angle(goal, obs_pixels, curr)
    
        print("Goal: {}".format(goal))
        print("Angle: {} (from [-1, 1])".format(angle))
        
        
    #create array of obstacles and goal estimation for dwa
    # get angle from dwa
    # turn wheels left/right according to angle from dwa
    cv2.imshow('img', img)
    cv2.waitKey(1)

def main():
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
                do_for_frame(img, obj_det, dwa)
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


def test_with_image():
    dwa = DWA()
    obj_det = yolo.create()
    path = os.path.join('api', 'projective_transformation', 'images', 'img2.jpg')
    img = cv2.imread(path)
    do_for_frame(img, obj_det, dwa)

main()

# test_with_image()
