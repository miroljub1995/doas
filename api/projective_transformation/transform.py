from __future__ import division
import os
import cv2
import numpy as np
from time import time
import sys

def find_line_coef(p1, p2):
    k1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    n1 = p1[1] - k1 * p1[0]
    return (k1, n1)

def find_x_on_line(line, y):
    k, n = line
    return int((y - n) // k)

def find_y_on_line(line, x):
    k, n = line
    return int(k * x + n)

def generate_trapesoid(intersec):
    left_edge_line = find_line_coef(intersec, (0, 60))
    right_edge_line = find_line_coef(intersec, (416, 60))

    top_x = 45
    p1 = (find_x_on_line(left_edge_line, top_x), top_x)
    p2 = (find_x_on_line(left_edge_line, 416), 416)
    p3 = (find_x_on_line(right_edge_line, 416), 416)
    p4 = (find_x_on_line(right_edge_line, top_x), top_x)

    return (p1, p2, p3, p4)

def get_intersection():
    point11 = (1, 94)
    point12 = (140, 51)
    point21 = (271, 48)
    point22 = (413, 106)

    #y = k * x + n
    k1, n1 = find_line_coef(point11, point12)
    k2, n2 = find_line_coef(point21, point22)

    intersect_x = int((n2 - n1) / (k1 - k2))
    intersect_y = int(k1 * intersect_x + n1)
    return (intersect_x, intersect_y)

def get_projective_transform(dst_size):
    intersec = get_intersection()
    trap = generate_trapesoid(intersec)
    dst_w, dst_h = dst_size
    m = cv2.getPerspectiveTransform(np.array(trap, dtype=np.float32), np.array([(0, 0), (0, dst_h), (dst_w, dst_h), (dst_w, 0)], dtype=np.float32))
    return m

def get_perspective_coordinates(coords, dst_size):
    if len(coords) == 0:
        return coords
    coords = np.reshape(coords, (-1, 1, 2))
    m = get_projective_transform(dst_size)
    dst = cv2.perspectiveTransform(coords, m)
    dst = np.reshape(dst, (-1, 4))
    return dst

def get_wrapped_img(img, dst_size):
    m = get_projective_transform(dst_size)
    return cv2.warpPerspective(img, m, dst_size)

def rects_to_pixels(rects, dst_size):
    bw_img = np.ones((dst_size[1], dst_size[0], 1))
    for obs in rects:
        x1, y1, x2, y2 = obs
        px1, py1, px2, py2 = x1, y2, x2, y2
        cv2.rectangle(bw_img, (px1, py1), (px2, py2), 0, 1)
    #cv2.imshow('test', bw_img)
    #cv2.waitKey(1)
    pixels = []
    for i, row in enumerate(bw_img):
        for j, p in enumerate(row):
            if p == 0.0:
                pixels.append((j, i))
    return pixels

# path = os.path.join(os.path.dirname(__file__), 'images', 'img4.jpg')
# img = cv2.imread(path)
# cv2.imshow('origin', img)
# # y = yolo.create()
# # out = y.detect(img)


# intersec = get_intersection()
# img = cv2.circle(img, intersec, 3, (0, 255, 0), 2)

# trap = generate_trapesoid(intersec)
# cv2.polylines(img, [np.array(trap).astype(np.int32)], True, (255, 0, 0), 1)
# cv2.imshow('polylines', img)


# dst_w, dst_h = (300, 500)
# H = cv2.getPerspectiveTransform(np.array(trap, dtype=np.float32), np.array([(0, 0), (0, dst_h), (dst_w, dst_h), (dst_w, 0)], dtype=np.float32))

# print(H)
# img = cv2.warpPerspective(img, H, (dst_w, dst_h))

# cv2.imshow('perspective', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
