from time import time
import math
import traceback
import os

import cv2
import numpy as np
from time import time

from api.camera_reader.reader import CameraReader
from api.dwa.dwa import DWA
from api.yolo import yolo
from api.projective_transformation import transform
# from api.car.servo import Servo

perspective_size = (500, 800)

def draw_pixels(img, pixels):
    for pix in pixels:
        x, y = pix
        cv2.circle(img, (x, y), 1, 255, 1)

def classify(img, lines):
    # alpha = 2 # Contrast control (1.0-3.0)
    # beta = -50 # Brightness control (0-100)
    # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = 0.33
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny = cv2.Canny(gray, lower, upper)

    l, r, c = [], [], []
    h, w, _ = img.shape
    for line in lines:
        mask = np.zeros((h, w, 1), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, line)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        canny_masked = cv2.bitwise_and(canny, canny, mask=mask)
        min_line_length = 50
        max_line_gap = 500
        h_lines = cv2.HoughLinesP(canny_masked, 1, np.pi/180, 15, None, min_line_length, max_line_gap)
        if h_lines is None:
            h_lines = []
        else:
            h_lines = h_lines.reshape((-1, 4))
        params = []
        for h_line in h_lines:
            h_x1, h_y1, h_x2, h_y2 = h_line
            if h_x1 == h_x2:
                continue
            h_params = np.polyfit((h_x1, h_x2), (h_y1, h_y2), 1)
            params.append(h_params)
        if len(params) > 0:
            params_avg = np.average(params, axis=0)
            slope_avg, y_intercept_avg = params_avg
            l_y1 = y1
            l_y2 = y2
            l_x1 = int((l_y1 - y_intercept_avg) / slope_avg)
            l_x2 = int((l_y2 - y_intercept_avg) / slope_avg)
            if slope_avg < 0:# < -0.5 or something less
                l.append(np.array([l_x1, l_y1, l_x2, l_y2], dtype=np.int32))
            elif slope_avg > 0:# > 0.5 or something more
                r.append(np.array([l_x1, l_y1, l_x2, l_y2], dtype=np.int32))
            else:
                c.append(np.array([l_x1, l_y1, l_x2, l_y2], dtype=np.int32))
    return np.array([l, r, c])

def do_for_frame(img, obj_detector, dwa, servo):
    img = cv2.resize(img, (416, 416))
    objs, _, labels = obj_detector.detect(img)
    print(objs)
    lines, obstacles = yolo.lines_and_obstacles(objs, labels)
    print('*************************')
    print("Lines: {}".format(lines))
    #print("Obstacles: {}".format(obstacles))

    #obstacles in perspective
    obst_bottom = [[x1, y2, x2, y2] for x1, y1, x2, y2 in obstacles]
    obst_pers_bottom = transform.get_perspective_coordinates(obst_bottom, perspective_size)
    print("Perspective obstacles: {}".format(obst_pers_bottom))

    #lines in perspective
    left, right, c = classify(img, lines)
    left_pers = transform.get_perspective_coordinates(left, perspective_size)
    right_pers = transform.get_perspective_coordinates(right, perspective_size)
    c_pers = transform.get_perspective_coordinates(c, perspective_size)

    #current car position
    curr_pos_pers = (int(perspective_size[0] / 2), perspective_size[1] - 1)
    angle = 0.0

    if len(c_pers) > 0:
        #line accross camera detected, need to turn left
        pass
    elif len(left_pers) > 0 and len(right_pers) > 0:
        # calculate goal in the middle of lines
        lx1, ly1, lx2, ly2 = map(int, left_pers[0])
        rx1, ry1, rx2, ry2 = map(int, right_pers[0])

        print("Line distance: {}".format(lx1 - rx1))
        
        mx1 = int((lx1 + rx1) / 2)
        my1 = int((ly1 + ry1) / 2)
        mx2 = int((lx2 + rx2) / 2)
        my2 = int((ly2 + ry2) / 2)

        print("Goal vars: {}".format((mx1, my1, mx2, my2)))

        k = float(my2 - my1) / (mx2 - mx1)
        gy = 10
        gx = int((gy - my1 + mx1 * k) / k)
        print("Goal: {}".format((gx, gy)))
        test_image = np.zeros((perspective_size[1], perspective_size[0], 1), dtype=np.uint8)


        print("All lines: {}, {}".format(obst_pers_bottom, left_pers))
        all_lines = np.vstack((obst_pers_bottom, left_pers))
        all_lines = np.vstack((all_lines, right_pers))
        all_lines_px = transform.lines_to_pixels(all_lines, perspective_size)

        angle = dwa.calculate_angle((gx, gy), all_lines_px, curr_pos_pers)
        print("Angle: {}".format(angle))
        # servo.set_from_range(angle, [-1, 1])

        draw_pixels(test_image, all_lines_px)
        cv2.imshow("DWA obstacles and lines", test_image)
    elif len(left_pers) > 0:
        pass
    elif len(right_pers) > 0:
        pass
    else:
        pass
        
    #create array of obstacles and goal estimation for dwa
    # get angle from dwa
    # turn wheels left/right according to angle from dwa
    obj_detector.draw(img, [objs, _, labels])
    cv2.imshow('img', img)
    cv2.waitKey(0)

def main():
    reader = None
    try:
        obj_det = yolo.create()
        dwa = DWA()
        # servo = Servo()
        servo = None
        reader = CameraReader()
        # reader = cv2.VideoCapture(0)

        while True:
            start = time()
            try: # not to brake program if one frame fail
                ret, img = reader.read()
                #print("Read time: {}".format(time() - start))
                do_for_frame(img, obj_det, dwa, servo)
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
    # servo = Servo()
    servo = None
    dwa = DWA()
    obj_det = yolo.create()
    path = os.path.join('api', 'projective_transformation', 'images', 'img3.jpg')
    img = cv2.imread(path)
    do_for_frame(img, obj_det, dwa, servo)

# main()

test_with_image()
