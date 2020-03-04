import cv2
from api.yolo import yolo
from api.camera_distance.distance import calculate_fl_obstacle , obstacle_distance

#print(calculate_fl_obstacle())
#exit()

cam=cv2.VideoCapture(0)
ret,img=cam.read()
y = yolo.create()
img = cv2.resize(img, (416, 416))
out = y.detect(img)
#y1, x1, y2, x2 = out[0][0]
# print(obstacle_distance(img, out[0]))
