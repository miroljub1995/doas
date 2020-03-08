import cv2

cap = cv2.VideoCapture(0)
ret, img = cap.read()
img = cv2.resize(img, (416, 416))
cv2.imwrite('saved_img.jpg', img)
