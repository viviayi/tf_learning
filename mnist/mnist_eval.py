# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:13:35 2019

@author: 11104510
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
                tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(
                tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        #测试时直接前向传播计算输出，不需正则化
        y = mnist_inference.inference(x, None)
        
        #计算正确率，使用arg_max得到类别
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        #通过变量重命名的方式加载模型
        variable_averages = tf.train.ExponentialMovingAverage(
                mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        #每隔EVAL_INTERVAL_SECS调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                        mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,
                                              feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g." % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found!")
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    evaluate(mnist)
    
if __name__ == '__main__':
    tf.app.run()