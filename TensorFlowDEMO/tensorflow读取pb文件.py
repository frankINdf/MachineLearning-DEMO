# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:30:07 2018

@author: Administrator
"""
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
sess = tf.Session()
with gfile.FastGFile('D:/MMNIST/modelpb.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def) # 导入计算图

# 初始化的过程    
sess.run(tf.global_variables_initializer())

# 需要先复原变量
print(sess.run('b1:0'))
## 1
#
## 输入
input_x = sess.graph.get_tensor_by_name('x:0')
print(input_x)
#input_y = sess.graph.get_tensor_by_name('y:0')
#
op = sess.graph.get_tensor_by_name('op_to_store:0')
print(op)
#ret = sess.run(op,  feed_dict={input_x: 5, input_y: 5})
#print(ret)
## 输出 26
