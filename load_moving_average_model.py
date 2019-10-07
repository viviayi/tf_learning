#!/usr/bin/env python
# coding: utf-8
'''
加载滑动平均模型并重命名变量，将平均后的影子变量映射到自身
'''
# In[1]:


import tensorflow as tf


# In[2]:


v = tf.Variable(0, dtype=tf.float32, name="v")
#saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
#使用滑动平均的variables_to_restore函数代替变量重命名
ema = tf.train.ExponentialMovingAverage(0.99)
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "model/moving_average_model.ckpt")
    print(sess.run(v))

