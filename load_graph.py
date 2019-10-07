# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:29:51 2019

@author: 11104510
"""
#注意要重启kernel才能运行成功，玄学
import tensorflow as tf

#使用和保存模型代码中一样的方式声明变量
#v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
#v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
#加载模型时可以给变量重命名，但是在saver中要指明对应的变量
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
result = v1 + v2

saver = tf.train.Saver({"v1":v1,"v2":v2})

with tf.Session() as sess:
    saver.restore(sess, "model/model.ckpt")
    print(sess.run(result))
    
"""

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "model/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
 """