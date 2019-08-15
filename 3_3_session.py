# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:24:14 2019

@author: 11104510
"""

import tensorflow as tf

#张量的定义
a = tf.constant([1, 2], name="a", dtype=tf.float32)
b = tf.constant([2.0, 3.0], name="b", dtype=tf.float32)
result = tf.add(a, b, name="add")

#会话执行运算
#不使用默认会话，则通过run得到计算值
with tf.Session() as sess1:
    print(sess1.run(result))
    
#使用默认会话，用eval得到张量值
sess2 = tf.Session()
with sess2.as_default():
    print(result.eval())
#自动将生成的会话注册为默认会话
sess3 = tf.InteractiveSession()
print(result.eval())
sess3.close()

#通过ConfigProto Protocol Buffer配置会话
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)
#allow_soft_placement=True 使运算无法被GPU支持时可以自动调整到CPU上，并且可以让其在拥有不同数量GPU上顺利运行
#log_device_placement=True 记录每个节点被安排在哪个设备上，方便调试，部署到生产环境中可设置为False