# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:50:04 2019

@author: 11104510
"""

import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

#通过tf.get_variables来获取变量。训练时创建，测试时加载取值，且滑动平均变量会自动重命名
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
            "weights", shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    #当给出了正则化生成函数时，将当前变量的正则化损失加入losses集合，使用add_to_collection
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

#定义网络前向传播过程
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
                [INPUT_NODE,LAYER1_NODE], regularizer)
        biases = tf.get_variable(
                "biases", [LAYER1_NODE],
                initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
                [LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable(
                "biases", [OUTPUT_NODE],
                initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
        
    return layer2
        