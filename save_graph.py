# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 19:37:49 2019

@author: 11104510
"""
#注意要重启kernel才能运行成功，玄学，否则会报错
import tensorflow as tf

#声明两个变量并计算他们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
#声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    #sess.run(result)
    #将模型保存
    saver.save(sess, "model/model.ckpt")
"""

v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "model/model.ckpt")
  print("Model saved in path: %s" % save_path)
"""