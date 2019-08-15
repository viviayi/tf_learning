# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:00:46 2019

@author: 11104510
"""

import tensorflow as tf

#通过不同的图来隔离张量
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable(
            "v", shape=[1], initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable(
            "v", shape=[1], initializer=tf.ones_initializer)

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
        
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
        
        