# -*- coding: utf-8 -*-

import tensorflow as tf
import os


class nn_model(object):
    def __init__(self):
        items = os.listdir()
        max_one = 0
        for item in items:
            if item.startswith('suprised_model'):
                cur_num = item[20:25]
                if int(cur_num) > int(max_one):
                    max_one = cur_num
        self.meta = '.\suprised_model.ckpt-' + max_one + '.meta'
        self.ckpt = '.\suprised_model.ckpt-' + max_one

    def predict_value(self, x_array):
        g1 = tf.Graph()  # 调用CNN网络
        sess = tf.Session(graph=g1)
        with sess.as_default():
            with sess.graph.as_default():
                saver = tf.train.import_meta_graph(self.meta)  # 载入图结构，保存在.meta文件中？
                saver.restore(sess, self.ckpt)  # 恢复模型？
                graph = tf.get_default_graph()
                input_shape = graph.get_operation_by_name('x_place').outputs[0]
                output_shape = graph.get_tensor_by_name("logits_eval:0")
                result = sess.run(output_shape, feed_dict={input_shape: x_array})
                # result = result[:, np.newaxis]
                result = result * 1e4  # 别忘了反归一化！！
                return result

