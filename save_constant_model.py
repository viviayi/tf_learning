#!/usr/bin/env python
# coding: utf-8

'''
将计算图中的变量及值用常量保存到模型中，并去除不必要的节点
'''
# In[1]:


import tensorflow as tf
from tensorflow.python.framework import graph_util


# In[2]:


v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2


# In[3]:


init_op = tf.global_variables_initializer()


# In[4]:


with tf.Session() as sess:
    sess.run(init_op)
    #导出当前计算图的graphdef部分，只需要这一部分即可完成输入到输出的计算
    graph_def = tf.get_default_graph().as_graph_def()
    
    #将图中的变量及取值转化为常量，同时将不必要的节点去掉
    #下面代码中最后的参数add给出了需要保存的节点名称
    #后面有零是节点输出的张量名称add:0
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    #将模型保存到文件
    with tf.gfile.GFile("model/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())

