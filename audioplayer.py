
import sys
import time
from naoqi import ALProxy


IP = "192.168.0.104"  # Replace here with your NaoQi's IP address.
PORT = 9559
try:
    aup = ALProxy("ALAudioPlayer", IP, PORT)
except Exception as e:
    print("Could not create proxy to ALAudioPlayer")
    print ("Error was: "+e)
    sys.exit(1)

#plays a file and get the current position 5 seconds later
fileId = aup.post.playFile("file_example_WAV_1MG.wav")
# print(aup.getVolume())
# fileId = aup.post.playFile("/usr/share/naoqi/wav/filename.wav")
time.sleep(5)
print(aup.getVolume(fileId))
#currentPos should be near 5 secs
# aup.play(fileId)
# currentPos = aup.getCurrentPosition(fileId)