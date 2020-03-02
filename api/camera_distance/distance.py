import cv2
from api.yolo import yolo


def calculate_fl_obstacle():
    fLSum = 0
    count = 0
    y = yolo.create()
    KNOWN_WIDTH = 10
    cam=cv2.VideoCapture(0)
    while True:
        inpt = raw_input("Do you want to calculate one more times? (y/n): ")
        if inpt == "n":
            break
        ret, img=cam.read()
        cv2.imshow('img', img)
        cv2.waitKey(0)
        while raw_input("Discard image? (y/n): ") == "y":
            ret, img=cam.read()
            cv2.imshow('img', img)
            cv2.waitKey(0)

        KNOWN_DISTANCE = input("Enter the distance: ")
        if ret==True:
            img = cv2.resize(img, (416, 416))

            #cv2.imwrite("image1.jpg", img)
            out = y.detect_image_clean(img)
            if len(out[0]) == 0:
                print("Did not detect anything")
                #cv2.imshow('img', img)
                #cv2.waitKey(0)
                continue
            #cv2.imwrite("image.jpg", img)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)
            print(out)
            y1, x1, y2, x2 = out[0][0]
            imgWithRect = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #cv2.imshow('img', imgWithRect)
            #cv2.waitKey(0)


            w = x2 - x1
            #w=w*0.0264583333
            foclLenght=(w* KNOWN_DISTANCE) / KNOWN_WIDTH
            print(w)
            print(foclLenght)
            fLSum += foclLenght
            count += 1
    cam.release()
    cv2.destroyAllWindows()
    return fLSum / count

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the image to the camera
    return (knownWidth * focalLength) / perWidth

def obstacle_distance(img, rectangles):
    KNOWN_WIDTH = 10
    print(rectangles)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    distances = []
    for (y1, x1, y2, x2) in rectangles:
        w = x2 - x1
        #w=w*0.0264583333
        foclLenght=580
        distance=distance_to_camera(KNOWN_WIDTH,foclLenght,w)
        distances.append(distance)
    return distances
