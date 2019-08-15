# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:24:44 2019

@author: 11104510
"""

import tensorflow as tf

#声明w1, w2两个变量，通过seed参数设定随机种子，保证每次运行得到的结果相同
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1), name="w2")

#暂时将输入定义为一个常量,x为1*2的矩阵，两个[[]]
#x = tf.constant([[0.7, 0.9]])

#使用placeholder定义位置，运行时再传入数据，batch为3，特征数为2时
x = tf.placeholder(tf.float32, shape=(3,2), name="input")

#获得输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
#将y转换为0-1，表示预测为正的概率
y = tf.sigmoid(y)
sess = tf.Session()

#w1,w2初始化
#sess.run(w1.initializer)
#sess.run(w2.initializer)

#所有变量一起初始化
init_op = tf.global_variables_initializer()
sess.run(init_op)

#print(sess.run(y))
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5,0.9]]}))
sess.close()