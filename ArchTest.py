# Matthew Steed
# Testing architecture for learning to play NES with MNIST

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def plot_list(x, y, problem, learning_rate):
  plt.plot(x, y)
  #ax.set_xticks(y)
  plt.ylabel('Validation Accuracy')
  plt.xlabel('Number of Epochs')
  plt.title('Problem {}, Learning rate = {}'.format(problem, learning_rate))
  plt.savefig('p{}lr{}.png'.format(problem,learning_rate), dpi=200) 
  plt.clf()
  plt.cla()
  plt.close()


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='VALID')

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')





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
#  h_pool1 = max_pool(h_conv1)

  W_conv2 = weight_variable([4,4,16,32])
  b_conv2 = bias_variable([32])

  h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
#  h_pool2 = max_pool(h_conv2)

  """
  Wconv3 = weight_variable([1,1,64,128])
  bconv3 = bias_variable([128])

  h_conv3 = tf.nn.relu(conv2d(h_pool2, Wconv3) + bconv3)
  h_pool3 = max_pool(h_conv3)
  """

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
  val_freq = 5000
  count = 0
  with tf.Session() as sess:
    N = TrainX.shape[0]
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
      for j in range(int(N/batch_size)):
        count += batch_size 
        idx = np.random.choice(N, batch_size)
        train_step.run(feed_dict={x_img:TrainX[idx], y_:TrainY[idx]})

        if (count % val_freq < batch_size):
          val_error = 1 - accuracy.eval(feed_dict={x_img:TestX, y_:TestY})
          epoch_num = count / N
          print("Validation loss after {} epochs: {}".format(epoch_num, 100*val_error))
          accuracy_list.append(100*val_error)
          x_axis_list.append(epoch_num)

    conv_array = W_conv1.eval()
    feat_map = h_conv1.eval(feed_dict={x_img:TrainX[[0]],y_:TrainY[[0]]})

  plot_list(x_axis_list, accuracy_list, 3, learning_rate)

  fig = plt.figure()
  for i in range(16):
    fig.add_subplot(4,4,i+1)
    plt.axis('off')
    plt.imshow(conv_array[:,:,0,i], cmap=plt.get_cmap('gray'))
   
  plt.savefig('conv_p3_lr{}.png'.format(learning_rate))
  plt.clf()
  plt.cla()
  plt.close()


  fig = plt.figure()
  for i in range(16):
    fig.add_subplot(4,4,i+1)
    plt.axis('off')
    plt.imshow(feat_map[0,:,:,i], cmap=plt.get_cmap('gray'))
   
  plt.savefig('feat_p3_lr{}.png'.format(learning_rate))
  plt.clf()
  plt.cla()
  plt.close()







def main():
  # Load data
  print("Loading Data:")
  TrainDigitX = np.loadtxt('TrainDigitX.csv', delimiter=',')
  TrainDigitY = np.loadtxt('TrainDigitY.csv', delimiter=',')
  TestDigitX = np.loadtxt('TestDigitX.csv', delimiter=',')
  TestDigitY = np.loadtxt('TestDigitY.csv', delimiter=',')

  # Convert labels to one hot vectors
  TrainDigitY = np.eye(10)[TrainDigitY.astype(int)]
  TestDigitY = np.eye(10)[TestDigitY.astype(int)]
  


  print("Training")
  create_network(TrainDigitX, TrainDigitY, TestDigitX, TestDigitY, learning_rate=0.001)




if __name__=='__main__':
  main()
