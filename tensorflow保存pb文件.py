

from __future__ import print_function
import os
# Import MNIST data
from tensorflow.python.framework import graph_util
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
logs_path = os.getcwd()
import tensorflow as tf
tf.reset_default_graph()
# Parameters
learning_rate = 0.1
num_steps = 10
batch_size = 128
display_step = 20

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input],name = 'x')
Y = tf.placeholder("float", [None, num_classes], name = 'y')

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1]),name = 'h1_w'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name = 'h2_w'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]),name = 'out_w')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]),name = 'b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]),name = 'b2'),
    'out': tf.Variable(tf.random_normal([num_classes]),name = 'out')
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name = 'op_to_store')

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,['op_to_store'])    
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        _,c=sess.run([train_op,loss_op], feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    with tf.gfile.FastGFile('D:/MMNIST/modelpb.pb',mode = 'wb') as f:
        f.write(constant_graph.SerializeToString())
    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
