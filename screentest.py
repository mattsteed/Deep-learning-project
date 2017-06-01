# Test for screenshot speed

#import PIL
from PIL import ImageGrab, Image
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.misc


def get_screen(box, i):
  img = ImageGrab.grab(box)
  img = img.convert("L")
  #img.save("./{}.png".format(i))
  return img


def display(image):
  plt.imshow(image, cmap='gray')
  plt.show()

screenshots = []
num_screens = 20

start_time = time.time()
for i in range(num_screens):
  img = get_screen((2*147,2*177,2*627,2*627), i)
  array = np.array(img)
  print(array.shape)
  array = scipy.misc.imresize(array, (84, 90), interp='nearest')
  array = array[:, 0:84]
  print(array.shape)
  array = array / np.max(array)
  screenshots.append(array)

end_time = time.time()
print("Time for {} screens: {}".format(num_screens, end_time - start_time))
print("Screens per second: {}".format(num_screens / (end_time - start_time)))

for i in range(len(screenshots)):
  plt.imshow(screenshots[i], cmap='gray')
  plt.show()



default_box = (2*147,2*177,2*627,2*627)

def take_screenshot(box=(2*147,2*177,2*627,2*627)):
  img = ImageGrab.grab(box)
  img = img.convert("L")
  array = np.array(img)
  array = scipy.misc.imresize(array, (84, 90), interp='nearest')
  array = array[:, 0:84]
  array = array / np.max(array)
  return array



from scipy import misc
r = misc.imread('test2.png', mode = 'L')
test = np.array(r)
test = scipy.misc.imresize(test, (84, 90), interp='nearest')
test = test[:, 0:84]
test = test / np.max(test)