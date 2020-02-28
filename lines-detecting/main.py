import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(os.path.join('input', 'Video_1.mp4'))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

if not os.path.exists('out'):
    os.makedirs('out')

outSobel = cv2.VideoWriter(os.path.join('out','sobel.mp4'), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
outCanny = cv2.VideoWriter(os.path.join('out','canny.mp4'), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
outHough = cv2.VideoWriter(os.path.join('out','hough.mp4'), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ###################################### Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)  # x
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)  # y
    gradient_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    sobel = gradient_magnitude.astype(np.uint8)

    bgrSobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    outSobel.write(bgrSobel)

    ###################################### Canny
    canny = cv2.Canny(gray,100,200)
    bgrCanny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    outCanny.write(bgrCanny)

    ##################################### Hough lines
    destImg = frame.copy()
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(destImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    outHough.write(destImg)

cap.release()
outSobel.release()
outCanny.release()
outHough.release()
cv2.destroyAllWindows()
