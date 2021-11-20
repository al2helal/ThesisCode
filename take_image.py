# -*- encoding: UTF-8 -*-
# Get an image from NAO. Display it and save it using PIL.

import sys
import time

# Python Image Library
from PIL import Image

from naoqi import ALProxy


def showNaoImage(IP, PORT):
  """
  First get an image from Nao, then show it on the screen with PIL.
  """

  camProxy = ALProxy("ALVideoDevice", IP, PORT)
  faceProxy = ALProxy("ALFaceDetection", IP, PORT)
  memoryProxy = ALProxy("ALMemory", IP, PORT)
  resolution = 2    # VGA
  colorSpace = 11   # RGB
  period = 500
  # faceProxy.subscribe("Test_Face", period, 0.0 )
  memValue = "FaceDetected"
  for i in range(0, 20):
    videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)

    time.sleep(0.5)
    val = memoryProxy.getData(memValue)
    if(val and isinstance(val, list) and len(val) >= 2):
      print("face detected")
      t0 = time.time()

      # Get a camera image.
      # image[6] contains the image data passed as an array of ASCII chars.
      naoImage = camProxy.getImageRemote(videoClient)
      if (naoImage == None):
        continue
      t1 = time.time()

      # Time the image transfer.
    #   print(f'acquisition delay , {t1 - t0}')

      camProxy.unsubscribe(videoClient)


      # Now we work with the image returned and save it as a PNG  using ImageDraw
      # package.

      # Get the image size and pixel array.
      imageWidth = naoImage[0]
      imageHeight = naoImage[1]
      array = naoImage[6]

      # Create a PIL Image from our pixel array.
      im = Image.frombytes("RGB", (imageWidth, imageHeight), array)

      # Save the image.
      im.save("Image/camImage_"+str(i)+".png", "PNG")

      im.show()



if __name__ == '__main__':
  IP = "192.168.0.104"  # Replace here with your NaoQi's IP address.
  PORT = 9559

  # Read IP address from first argument if any.
  if len(sys.argv) > 1:
    IP = sys.argv[1]

  naoImage = showNaoImage(IP, PORT)