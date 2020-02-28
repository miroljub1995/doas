import cv2
from api.yolo import yolo


def calculate_fl_obstacle():
    y = yolo.create()
    KNOWN_WIDTH = 10
    cam=cv2.VideoCapture(0)
    ret,img=cam.read()
    KNOWN_DISTANCE = input("Enter the distance")
    if ret==True:
        out = y.detect_image_clean(img)
        print(out)
        y1, x1, y2, x2 = out[0][0]
        w = x2 - x1
        w=w*0.0264583333
        foclLenght=(w* KNOWN_DISTANCE) / KNOWN_WIDTH
        print(w)
        print(foclLenght)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the image to the camera
    return (knownWidth * focalLength) / perWidth

def obstacle_distance(img, rectangles):
    KNOWN_WIDTH = 10
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    distances = []
    for (y1, x1, y2, x2) in rectangles:
        w = x2 - x1
        w=w*0.0264583333
        foclLenght=0.001 # precalculated
        distance=distance_to_camera(KNOWN_WIDTH,foclLenght,w)
        distances.append(distance)
    return distances