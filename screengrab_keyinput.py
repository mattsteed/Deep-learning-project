__author__ = 'jackliang'

from PIL import ImageGrab
import os
import time

ims = []
start_time = time.time()
for i in range(0, 20):
    ims = ims + [ImageGrab.grab(bbox=(0, 40, 2 * 256, 2 * (256 + 40)))]

print((time.time() - start_time))

for i in range(0, 20):
    ims[i].show()


## an example of how to open nestopia, make mario jump
## I have a binded to q, b binded to w
## up, left, down, right binded to i j k l respectively
import time
import os
import pykeyboard
import numpy as np


def init():
    cmd = """
    osascript -e 'tell application \"Nestopia\" to open file \"Users:jackliang 1:Desktop:SMB.nes\"
    '
    """

    time.sleep(.02)

    os.system(cmd)
    cmd = """
    osascript -e 'tell application "Nestopia"
        activate
    end tell'
    """

    os.system(cmd)

def input_command(keys, sleep_time, lag_param):
    cmd = """
    osascript -e 'tell application "Nestopia"
        activate
    end tell'
    """

    os.system(cmd)

    time.sleep(lag_param)

    k = pykeyboard.PyKeyboard()
    for key in keys:
        k.press_key(key)

    time.sleep(sleep_time)

    for key in keys:
        k.release_key(key)

def load_state(filename):
    init()

    cmd = """
    osascript -e 'tell application "System Events" to keystroke "d" using command down'
    """
    os.system(cmd)

    k = pykeyboard.PyKeyboard()
    for key in list(filename)[:5]:
        k.tap_key(key)

    cmd = """
    osascript -e 'tell application "System Events" to key code 36'
    """
    os.system(cmd)









test = ['q', 'l']
input_command(['z'], .1, 0.01)


load_state('12345.frz')
time.sleep(3)
input_command(test,3 ,0.01)
input_command(test,3 ,0.01)





