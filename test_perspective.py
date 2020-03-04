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

# reshaped = np.reshape(out[0], (-1, 1, 2))
# print(reshaped)

dst_size = (20, 40)
# img_wrapped = transform.get_wrapped_img(img, dst_size)
# cv2.imshow('img', img_wrapped)
# cv2.waitKey(0)
# exit()
res_rects = transform.get_perspective_coordinates(out[0], dst_size)
print(res_rects)
# res_rects = np.reshape(res_rects, (-1, 4))
# print(res_rects)

img_wrapped = transform.get_wrapped_img(img, dst_size)
for rect in zip(res_rects, out[2]):
    x1, y1, x2, y2 = rect[0]
    img_wrapped = cv2.rectangle(img_wrapped, (x1, y1), (x2, y2), colors[rect[1]], -1)

lines = []
for obj in zip(res_rects, out[2]):
    if obj[1] == 0:
        lines.append(obj[0])
print("---------------")
print(lines)

left = []
right = []
if lines[0][0] < lines[1][0] and lines[0][2] < lines[1][3]:
    left = lines[0]
    right = lines[1]
else:
    right = lines[0]
    left = lines[1]
    
print(left)
print(right)

l_y = (left[1] + left[3]) / 2
r_y = (right[1] + right[3]) / 2
middle = (int((left[2] + right[0]) / 2), int((l_y + r_y) / 2))

cv2.circle(img_wrapped, middle, 2, (255, 0, 0), 2)

cv2.imshow('img', img_wrapped)
cv2.waitKey(0)