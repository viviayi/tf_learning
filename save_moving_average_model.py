#!/usr/bin/env python
# coding: utf-8
'''
保存滑动平均模型
'''
# In[1]:


import tensorflow as tf


# In[2]:


v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.global_variables():
    print(variables.name)


# In[3]:


ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())


# In[4]:


for variables in tf.global_variables():
    print(variables.name)


# In[6]:


saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    saver.save(sess, "model/moving_average_model.ckpt")
    print(sess.run([v, ema.average(v)]))


# In[ ]:




