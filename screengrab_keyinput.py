__author__ = 'jackliang'


from PIL import ImageGrab
import os
import time
from getch import getch

ims = []
start_time = time.time()
for i in range(0,20):
    ims = ims + [ImageGrab.grab(bbox=(0,40,2*256,2*(256+40)))]

print((time.time() - start_time))


for i in range(0, 20):
    ims[i].show()




def keyspressed():
    start_time = time.time()
    keys = []
    while True:
        key = getch()
        keys = keys + [(key, time.time())]
        if key == '\n':
            print(str(time.time() - start_time))
            print(len(keys))
            return keys
        if time.time() - start_time > 10:
            return keys


