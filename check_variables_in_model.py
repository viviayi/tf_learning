#!/usr/bin/env python
# coding: utf-8
'''
查看模型中的变量和变量名
'''
# In[15]:


import tensorflow as tf


# In[26]:


reader = tf.train.NewCheckpointReader('model/model.ckpt')


# In[29]:


global_variables = reader.get_variable_to_shape_map()
for variable_name in global_variables:
    print(variable_name, global_variables[variable_name])


# In[30]:


print("Values for variable v1 is: ", reader.get_tensor("v1"))


# In[ ]:




