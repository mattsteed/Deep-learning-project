# Matthew Steed
# Architecture for learning to play NES

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import screentest
import screengrab_keyinput as sck
from copy import copy

class SC:
  def __init__(screenshot=None, save_state=None):
    self.screenshot = screenshot # 84x84 numpy array in grayscale
    self.save_state = save_state # string of filename for savestate

  def get_screenshot(self):
    return self.screenshot

  def get_save_state(self):
    return self.save_state

keys = {0:'q', 1:'w', 2:'j', 3:'i', 4:'k', 5:'l'}


from scipy import misc
r = misc.imread('reward.png', mode = 'L')
rwd = np.array(r)
rwd = scipy.misc.imresize(rwd, (84, 90), interp='nearest')
rwd = rwd[:, 0:84]
rwd = rwd / np.max(rwd)

f = misc.imread('failure.png', mode = 'L')
fail = np.array(f)
fail = scipy.misc.imresize(fail, (84, 90), interp='nearest')
fail = fail[:, 0:84]
fail = fail / np.max(fail)




def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='VALID')


def create_network(screenshots, num_steps=1000, learning_rate=0.001, gamma = 0.5):
  image_dim = 84
  seq_len = 4
  x_img = tf.placeholder(tf.float32, shape=[None, image_dim, image_dim, seq_len])
#  x_img = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 6])
#  x_img2 = tf.reshape(x_img, [-1, 28, 28, 1])


  W_conv1 = weight_variable([8,8,1,16])
  b_conv1 = bias_variable([16])

  h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1, 4) + b_conv1)

  W_conv2 = weight_variable([4,4,16,32])
  b_conv2 = bias_variable([32])

  h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

  new_size = int((image_dim/4 - 1)/2 - 1)

  W_fc1 = weight_variable([new_size*new_size*seq_len*32, 256])
  b_fc1 = bias_variable([256])

  h_conv2_flat = tf.reshape(h_conv2, [-1, new_size*new_size*seq_len*32])

  h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

  W_out = weight_variable([256, 6])
  b_out = bias_variable([6])

  y_out = tf.matmul(h_fc1, W_out) + b_out





  # Error step
  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                  labels=y_, logits=y_out))
  square_loss = tf.reduce_sum(tf.square(y_ - y_out))

  train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(square_loss)

  #correct = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
  #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

  #accuracy_list = []
  #x_axis_list = []
  #conv_array = None


  # Training step
  # INSERT Q LEARNING CODE OR FUNCTION HERE
  with tf.Session() as sess:
    for i in range(num_steps):  
      num_screens = len(screenshots)
      start_idx = np.random.randint(3, high=num_screens)
      x = np.zeros((1, 84, 84, 4))
      x[0,:,:,0] = screenshots[start_idx-3].get_screenshot()
      x[0,:,:,1] = screenshots[start_idx-2].get_screenshot()
      x[0,:,:,2] = screenshots[start_idx-1].get_screenshot()
      x[0,:,:,3] = screenshots[start_idx].get_screenshot()

      move = sess.run([y_out], feed_dict={x_img:x})
      label = copy(move).reshape(1, 6)
      two_hot = np.zeros(6)
      two_hot[np.argmax(move[0:2])] = 1
      two_hot[np.argmax(move[2:6] + 2)] = 1

      # make move in game using two_hot

      move_1 = np.argmax(move[0:2])
      move_2 = np.argmax(move[2:6] + 2)
      moves = [keys[move_1], keys[move_2]]

      sck.input_command(moves, 1./2.5, 0.01)

      # take screenshot and save state, save to new class instance
      new_state = SC()
      new_sc = screentest.take_screenshot() # argument should be (x, y, x', y')
      f_name = "{:05d}".format(31)[::-1] + "save.frz"
      #new_save = sck.save_state("{:05d}save".format(i) + '.frz')
      new_save = sck.save_state(f_name)




      # insert this instance into screenshots array at index start_idx+1
      screenshots.insert(start_idx+1, new_state)



      # Get reward if there should be a reward at this instance, store in 
      # reward_check variable

      reward_check = 0
      if np.linalg.norm(new_state.get_screenshot() - fail, 'fro') < 5.0: ## possibly tune these parameters
        reward_check = -1
      elif np.linalg.norm(new_state.get_screenshot() - rwd, 'fro') < 5.0:
        reward_check = 1

      right_bias = 0.1

      if (reward_check == -1 or reward_check == 1):
        label = reward_check * two_hot.reshape(1, 6)
        label[0][move_1] = reward_check
        label[0][move_2] = reward_check
        if (move_2 == 5):
          label[0][5] += right_bias
        train_step.run(feed_dict={x_img:x, y_:label}

      else: 
        x_bar = np.zeros((1, 84, 84, 4))
        x_bar[0,:,:,0] = screenshots[start_idx-2].get_screenshot()
        x_bar[0,:,:,1] = screenshots[start_idx-1].get_screenshot()
        x_bar[0,:,:,2] = screenshots[start_idx].get_screenshot()
        x_bar[0,:,:,3] = screenshots[start_idx+1].get_screenshot()
        new_label = sess.run([y_out], feed_dict={x_img:x_bar}).reshape(1,6)
        label[0][move_1] = gamma * new_label[0][move_1]
        label[0][move_2] = gamma * new_label[0][move_2]
        if (move_2 == 5):
          label[0][5] += right_bias
        train_step.run(feed_dict={x_img:x, y_:label}






def main():
  if __name__=='__main__':
    main()
