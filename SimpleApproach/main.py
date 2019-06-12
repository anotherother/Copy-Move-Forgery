import cv2
from SimpleApproach.DetectionCMF import DetectionCMF
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("./003_F.png" ,0)
    height, width= img.shape
    asd = DetectionCMF(img, height, width, 8,3.5,8,100,5)
    image = asd.detect_forgery()
    plt.imshow(image)
    plt.show()





