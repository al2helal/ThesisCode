# -*- encoding: UTF-8 -*-
# Get an image from NAO. Display it and save it using PIL.

import sys
import time

# Python Image Library
from PIL import Image

from naoqi import ALProxy
IP = "192.168.0.104"  # Replace here with your NaoQi's IP address.
PORT = 9559

def sad():
  """
  Say "Are you upset? What happened to you? Tell me, your mind will get better."
  """
  text = 'Are you upset? What happened to you? Tell me, your mind will get better.'
  tts = ALProxy("ALTextToSpeech", IP, PORT)
  tts.setParameter("speed", 0.5)
  tts.say('tomar ki Mon kharap? tomar ki hoeche amake bolo')

def angry():
  """
  Do nothing. Sit down so that child can't harm any of the robot.
  """
  try:
    postureProxy = ALProxy("ALRobotPosture", IP, 9559)
  except Exception as e:
    print('Could not create proxy to ALRobotPosture')
    print("Error was: "+e)
  postureProxy.goToPosture("SitRelax", 1.0)

def attention():
  """
  Say 'My name is NAO, I can walk, I can talk'
  """
  try:
    postureProxy = ALProxy("ALRobotPosture", IP, 9559)
    motionProxy = ALProxy("ALMotion", IP, 9559)
    tts = ALProxy("ALTextToSpeech", IP, PORT)
  except Exception as e:
    print('Could not create proxy to ALRobotPosture/ALTextToSpeech')
    print("Error was: "+e)
  pNames = "Body"
  pStiffnessLists = 1.0
  pTimeLists = 1.0
  motionProxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)
  id = postureProxy.goToPosture("Sit", 1.0)
  tts.say("I can sit")

  id = postureProxy.goToPosture("StandInit", 1.0)
  tts.say("I can stand")
  postureProxy.wait(id, 0)
  # tts.say('My name is NAO, I can speek')
  # motionProxy.moveInit()
  # id = motionProxy.post.moveTo(0.1, 0, 0)
  # tts.say("I can walk forward")
  # motionProxy.wait(id, 0)
  # id = motionProxy.post.moveTo(-0.1, 0, 0)
  # tts.say("I can walk backward")
  # motionProxy.wait(id, 0)
  # id = postureProxy.goToPosture("Sit", 1.0)
  # tts.say("I can sit")
  # postureProxy.wait(id, 0)
  # id = postureProxy.goToPosture("LyingBelly", 1.0)
  # tts.say("I can lying")
  # postureProxy.wait(id, 0)
  # id = postureProxy.goToPosture("LyingBack", 1.0)
  # tts.say("I can lying")
  # postureProxy.wait(id, 0)
 

if __name__ == '__main__':


  # Read IP address from first argument if any.
  if len(sys.argv) > 1:
    IP = sys.argv[1]

  # behavior_1 = sad()
  # behavior_2 = angry()
  behavior_3 = attention()
  # postureProxy = ALProxy("ALRobotPosture", IP, 9559)
  # print(postureProxy.getPostureList())
