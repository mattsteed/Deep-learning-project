# Test for screenshot speed

#import PIL
from PIL import ImageGrab, Image
import numpy as np
import matplotlib.pyplot as plt
import time


def get_screen(box, i):
  img = ImageGrab.grab(box)
  img = img.convert("L")
  #img.save("./{}.png".format(i))
  return img


def display(image):
  plt.imshow(image, cmap='gray')
  plt.show()

screenshots = []
num_screens = 50

start_time = time.time()
for i in range(num_screens):
  img = get_screen((2*267,2*289,2*567,2*589), i)
  array = np.array(img)
#  array = array / np.max(array)
  screenshots.append(array)

end_time = time.time()
print("Time for {} screens: {}".format(num_screens, end_time - start_time))
print("Screens per second: {}".format(num_screens / (end_time - start_time)))

for i in range(len(screenshots)):
  plt.imshow(screenshots[i], cmap='gray')
  plt.show()


