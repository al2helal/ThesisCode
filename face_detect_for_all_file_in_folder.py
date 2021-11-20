import cv2
import sys, os

# Get user supplied values
# imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
filelist = os.listdir('../Data/temp')
print(filelist)
os.chdir('../Data/temp')

for f in filelist:
    # Create the haar cascade

    # Read the image
    image = cv2.imread(f)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# closing all open windows
    cv2.destroyAllWindows()
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(48, 48),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        f1 = gray[y:y + h, x:x + w]
        f2 = cv2.resize(f1, (48, 48), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('../temp_gray/'+f, f2)

    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)
