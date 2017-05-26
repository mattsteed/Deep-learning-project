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
import os
import time

cmd1 = """osascript -e 'tell application "Nestopia"
    activate
    tell application "System Events" to keystroke "a"
end tell'
"""

start_time = time.time()
while True:
    os.system(cmd1)
    if time.time() - start_time > 5:
        break


import time
import os
import pykeyboard
import numpy as np

cmd = """
osascript -e 'tell application "Nestopia"
    activate
end tell'
"""

os.system(cmd)

time.sleep(.1)

k = pykeyboard.PyKeyboard()
k.press_key('q')
time.sleep(.5)
k.release_key('q')




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


test = ['a', 'l']
input_command(['z'], .1, 0.01)


time.sleep(0.5)
input_command(test,10 ,0.01)


