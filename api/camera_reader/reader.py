import cv2
import threading

class CameraReader (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(0)
        self.res = self.cap.read()
        self.running = True
        self.start()

    def run(self):
        while self.running:
            self.res = self.cap.read()

    def read(self):
        return self.res

    def release(self):
        self.running = False
        self.join()
        self.cap.release()

def main():
    reader = CameraReader()
    while True:
        ret, img = reader.read()
        cv2.imshow("img", img)
        cv2.waitKey(500)
    reader.release()
    cv2.destroyAllWindows()


class CameraReader1 (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.start()

    def run(self):
        while self.running:
            self.cap.grab()

    def read(self, image=None):
        return self.cap.retrieve(image=image)

    def release(self):
        self.running = False
        self.join()
        self.cap.release()

# main()