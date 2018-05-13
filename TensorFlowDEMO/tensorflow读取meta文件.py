# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:13:45 2018

@author: Administrator
"""
import os
import tensorflow  as tf
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow
tf.reset_default_graph()
meta_file_path = os.getcwd()

sess = tf.Session()
saver = tf.train.import_meta_graph('D:/MMNIST/model2.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('D:/MMNIST/'))
reader = tf.train.NewCheckpointReader('D:/MMNIST/model2.ckpt')  
  
variables = reader.get_variable_to_shape_map()  
  
for ele in variables:  
    print(ele)  
