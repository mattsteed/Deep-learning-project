# Matthew Steed
# Architecture for learning to play NES

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='VALID')




def create_network(TrainX, TrainY, TestX, TestY, \
             epochs=10, batch_size=50, learning_rate=0.001):

  image_dim = 28
  seq_len = 1
#  x_img = tf.placeholder(tf.float32, shape=[None, image_dim, image_dim, seq_len])
  x_img = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])
  x_img2 = tf.reshape(x_img, [-1, 28, 28, 1])


  W_conv1 = weight_variable([8,8,1,16])
  b_conv1 = bias_variable([16])

  h_conv1 = tf.nn.relu(conv2d(x_img2, W_conv1, 4) + b_conv1)

  W_conv2 = weight_variable([4,4,16,32])
  b_conv2 = bias_variable([32])

  h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

  new_size = int((image_dim/4 - 1)/2 - 1)

  W_fc1 = weight_variable([new_size*new_size*seq_len*32, 256])
  b_fc1 = bias_variable([256])

  h_conv2_flat = tf.reshape(h_conv2, [-1, new_size*new_size*seq_len*32])

  h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

  W_out = weight_variable([256, 10])
  b_out = bias_variable([10])

  y_out = tf.matmul(h_fc1, W_out) + b_out





  # Error step
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                  labels=y_, logits=y_out))

  train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

  correct = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

  accuracy_list = []
  x_axis_list = []
  conv_array = None


  # Training step
  # INSERT Q LEARNING CODE OR FUNCTION HERE






def main():



if __name__=='__main__':
  main()
