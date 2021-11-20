import cv2
import sys, os

f = '/home/alhelal/NAO/camImage.png'
image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
cv2.imwrite('/home/alhelal/NAO/camImage_gray.png', image)