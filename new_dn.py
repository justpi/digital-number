# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018/1/5/
    项目名称： 数字图片识别
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import timeit
import dn
from sklearn.model_selection import train_test_split

def main():
    dataSet = dn.loadData('./train.csv')
    labels, data = dn.sepData(dataSet)
    img, label = dn.load_tfrecord()
    img_test, label_test = dn.load_tfrecord(file_name='./test_img.tfrecords')
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    img_raw_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=32, capacity=100,
                                                        min_after_dequeue=16)
    img_test, label_test = tf.train.shuffle_batch([img_test, label_test], batch_size=32, capacity=100,min_after_dequeue=16)


    x = tf.placeholder('float', [None, 784])
    y_ = tf.placeholder('float', [None, 10])

    # 第一层
    W_conv1 = dn.weight_variable([5, 5, 1, 32])
    b_conv1 = dn.bias_variable([32])
    x_image = tf.reshape(x, shape=[-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(dn.conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = tf.nn.relu(dn.max_pool_2x2(h_conv1))
    # 第二层
    W_conv2 = dn.weight_variable([5, 5, 32, 64])
    b_conv2 = dn.bias_variable([64])
    h_conv2 = tf.nn.relu(dn.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = tf.nn.relu(dn.max_pool_2x2(h_conv2))
    # 全连接层
    W_fc = dn.weight_variable([7 * 7 * 64, 1024])
    b_fc = dn.bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, shape=[-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
    # dropout
    keep_prob = tf.placeholder('float')
    h_fc_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    W_fc2 = dn.weight_variable([1024, 10])
    b_fc2 = dn.bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)

    # 评估
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)
    corret_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(corret_prediction, dtype='float'))


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.initialize_all_variables())
        for epoch in range(20):
            for i in range(int(len(x_train)/32)):
                x_batch, y_batch = sess.run([img_raw_batch, label_batch])
                train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
                if i%100 == 0:
                    print("第%s个epoch,第%s个batch：训练准确率为%s" %(epoch, i, train_accuracy))
                train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
            for i in range(int(len(x_test)/32)):
                x_test, y_test = sess.run([img_test, label_test])
                if i%100 == 0:
                    print("第%s个epoch结束，训练结束，测试误差为%g" % (epoch, accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})))
        coord.request_stop()
        coord.join(threads)
        saver = tf.train.Saver()
        saver.save(sess, './model.ckpt', global_step=10)
        print("训练结束！")




if __name__ == '__main__':
    main()