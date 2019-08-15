# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:05:31 2019

@author: 11104510
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#print("Training data size: ", mnist.train.num_examples)
#
#print("Validating data size: ", mnist.validation.num_examples)
#
#print("Testing data size: ", mnist.test.num_examples)
#
#print("Example training data: ", mnist.train.images[0])
#
#print("Example training data label: ", mnist.train.labels[0])

#batch_size = 100
#xs, ys = mnist.train.next_batch(batch_size)
#print("X shape: ", xs.shape)
#print("Y shape: ", ys.shape)

#Mnist 数据集相关常数
INPUT_NODE = 784 #输入节点数，图像像素数28*28
OUTPUT_NODE = 10  #输出节点数，图像类别数
#配置神经网络参数
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001#正则化项在损失函数中的系数
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99#滑动平均衰减率

#辅助函数：计算前向传播的结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    '''
    计算前向传播的结果，参数分别是
    input_tensor: 输入数据（图像像素一维向量）
    avg_class: 滑动平均类
    weights1: 第一层的权重参数
    biases1: 第一层的偏置参数
    weights2: 第二层的权重参数
    biases2: 第二层的权重参数
    '''
    if avg_class == None:
        #计算第一层输出，使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        #计算输出层，损失函数会一并计算softmax，因此前向传播输出可不加
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(
                tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')
    
    #生成隐藏层参数，截断正态分布，产生的正态分布的值与均值的差不超过两倍标准差
    weights1 = tf.Variable(
            tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(
            tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
    #计算当前参数下的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)
    #定义训练轮数变量
    global_step = tf.Variable(0, trainable=False)
    #给定滑动平均衰减率和训练轮数，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    
    #在所有可训练变量上滑动
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #计算滑动平均之后的前向传播值
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    
    #计算交叉熵作为损失函数
    # tf.nn.sparse_softmax_cross_entropy_with_logits当分类问题中只有一个正确答案时，可以加速交叉熵的计算
    #y_是正确分类，我们的数据是长度为10的数组，函数参数需要正确类别序号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失
    loss = cross_entropy_mean + regularization
    
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples/BATCH_SIZE,  #训练完所有数据需要的迭代次数
            LEARNING_RATE_DECAY)
    
    #global_step自动
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #train_op = tf.group(train_step, variables_average_op)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    #检验结果是否正确
    #首先比较预测结果和label是否相同，然后将bool型转变为实数型计算平均
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        
        for i in range(TRAINING_STEPS):
            if i % 1000 ==0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g" % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))
        
def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)
    
if __name__ == '__main__':
    tf.app.run()
    
    
    
    
    
