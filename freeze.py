# -*- coding:utf-8 -*-
'''
固化银行卡模型时报错
'''
import tensorflow as tf
import os

modle_dir = "E:/tf_learning/card_number_detection_model"
checkpoint = tf.train.get_checkpoint_state(modle_dir)
input_checkpoint = checkpoint.model_checkpoint_path
print(input_checkpoint)

absolute_model = '/'.join(input_checkpoint.split('/')[:-1])
print(absolute_model)

output_grap = absolute_model + "/frozen_model.pb"
with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',clear_devices=True)
                                      
    saver.restore(sess, input_checkpoint)
    # 打印图中的变量，查看要保存的
    for op in tf.get_default_graph().get_operations():
        print(op.name, op.values())

    output_grap_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   tf.get_default_graph().as_graph_def(),
                                                                   output_node_names=['bbox_pred','cls_pred'])
    with tf.gfile.GFile(output_grap, 'wb') as f:
        f.write(output_grap_def.SerializeToString())
    print("%d ops in the final graph." % len(output_grap_def.node))