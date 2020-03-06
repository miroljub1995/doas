import cv2
from api.yolo import yolo
import os
from api.projective_transformation import transform
import numpy as np

#print(calculate_fl_obstacle())
#exit()

color = (255, 0, 0)

colors = [(0, 0, 255), (0, 255, 0)]

# img = cv2.imread(os.path.join('api', 'projective_transformation', 'images', 'image3941.jpg'))
img = cv2.imread(os.path.join('api', 'projective_transformation', 'images', 'image3607.jpg'))
y = yolo.create()
out = y.detect(img)
print(out[0])

#x1, y1, x2, y2 = out[0][0]
img_tmp = np.copy(img)
for rect in out[0]:
    x1, y1, x2, y2 = rect
    img_tmp = cv2.rectangle(img_tmp, (x1, y1), (x2, y2), color, 2)
cv2.imshow('img1', img_tmp)
cv2.waitKey(0)

dst_size = (5, 8)
res_rects = transform.get_perspective_coordinates(out[0], dst_size)
print(res_rects)

img_wrapped = transform.get_wrapped_img(img, dst_size)
for rect in zip(res_rects, out[2]):
    x1, y1, x2, y2 = rect[0]
    img_wrapped = cv2.rectangle(img_wrapped, (x1, y1), (x2, y2), colors[rect[1]], -1)

lines, obstacles = yolo.lines_and_obstacles(res_rects, out[2])
print("Detected ---------------")
print(lines)
print(obstacles)
print("------------------------")

###########################################
# getting pixels for obstacles (input to dwa)
bw_img = np.ones((dst_size[1], dst_size[0], 1))
for obs in obstacles:
    x1, y1, x2, y2 = obs
    cv2.rectangle(bw_img, (x1, y1), (x2, y2), 0, 1)
cv2.imshow('bw', bw_img)
cv2.waitKey(0)

pixels = []
print("Getting pixels")
for i, row in enumerate(bw_img):
    for j, p in enumerate(row):
        if p == 0.0:
            pixels.append((j, i))

    # if p == 0:
    #     pixels.append()
print(pixels, len(pixels))
print("Pixels done")
##############################

##############################
# calculate middle of lines (goal)
middle = yolo.calculate_goal(lines)
print("Middle: {}".format(middle))
##############################

cv2.circle(img_wrapped, middle, 0, (255, 0, 0), 1)

cv2.imshow('img', img_wrapped)
cv2.waitKey(0)