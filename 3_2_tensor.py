# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:54:03 2019

@author: 11104510
"""

import tensorflow as tf
#tf.constant 是一个计算，这个计算的结果为一个张量，保存在变量a中
#a,b是对常量生成这个运算结果的引用，后续使用时可以直接引用变量
#下面是直接计算向量和
#result = tf.constant([1.0, 2.0]) + tf.constant([2.0, 3.0])
a = tf.constant([1, 2], name="a", dtype=tf.float32)
b = tf.constant([2.0, 3.0], name="b", dtype=tf.float32)
result = tf.add(a, b, name="add")
print(result)

#通过图来指定运算设备
#g = tf.Graph()
#with g.device('/gpu:0'):
#    result = a + b