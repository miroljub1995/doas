import cv2
from api.yolo import yolo

y = yolo.create()
# img = Image.open("api/yolo/test_img.jpg")
img = cv2.imread("api/yolo/test_img.jpg")
out = y.detect_image_clean(img)

distances

print(out)