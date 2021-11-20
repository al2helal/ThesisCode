import cv2, os
dirs = os.listdir('video')
os.chdir('video')
count = 0
for d in dirs:
    vidcap = cv2.VideoCapture(d)
    success,image = vidcap.read()
    while success:
        cv2.imwrite("../video_to_images/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        print(f'image #{count}')
        count += 1