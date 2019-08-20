# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:12:21 2019

@author: 11104510
"""

import tensorflow as tf
from numpy.random import RandomState

######################################
###           网络部分              ###
######################################

#定义训练数据batch大小
batch_size = 8

#定义神经网络的参数，网络输入两个特征，隐藏层三个节点，输出一个概率
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name="w2")

#在输入数据的shape的一个维度上用None可以方便使用不同的batch大小，训练和测试的不同
x  = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

#定义前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和反向传播算法
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
        + (1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


######################################
###             数据部分            ###
######################################
#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
#定义x1+x2<1的为正样本，其余为负样本，正样本表示为1，负样本表示为0
Y = [[int(x1+x2<1)] for (x1, x2) in X]

saver = tf.train.Saver()
######################################
###             训练部分            ###
######################################
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #变量初始化
    sess.run(init_op)
    
    #训练之前的参数
    print(sess.run(w1))
    print(sess.run(w2))
    
    #设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        
        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                    cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps, cross entropy on all data is %g" % (i, total_cross_entropy))
    saver.save(sess, 'model/binary_classify.ckpt')        
    print(sess.run(w1))
    print(sess.run(w2))
    
    
    
    