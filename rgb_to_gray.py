import cv2
import sys, os

filelist = os.listdir('../Data/happy')
print(filelist)
os.chdir('../Data/happy')

for f in filelist:
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('../gray_happy/'+f, image)