import cv2
import numpy as np

class Money():
    def __init__(self) -> None:
        value = 0
        coin = False
        bnote = False
        image = None

    def find_coin(self, img):
        scaling = 800.0/max(img.shape[0:2])
        print(scaling)
        img_gray = cv2.resize(img, None, fx=scaling, fy=scaling)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.blur(img_gray, (5,5))
        coins = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.2, 30, param2 = 35, minRadius = 20, maxRadius = 50)
        coins = (np.round(coins[0,:]) / scaling).astype("int")
        self.coin = coins

    def find_bnote(self):
        pass

    def load_image(self, image):
        self.image = open(image)