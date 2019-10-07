#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[5]:


saver = tf.train.import_meta_graph("model/model.ckpt.meta")


# In[7]:


with tf.Session() as sess:
    saver.restore(sess, "model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

