from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
# Import data 手动导入到MNIST_data下
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def main(_):
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])  # 向量化的图片
    W = tf.Variable(tf.zeros([784, 10]))  # 权重矩阵
    b = tf.Variable(tf.zeros([10]))  # 偏置
    # hypothesis 假设的概率分布
    # y = tf.matmul(x, W) + b
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Define loss and optimizer 有效的概率分布/实际的概率分布
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,下面是交叉熵的原始公式
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable. 数值不稳定#
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    # labels: Each row labels[i] must be a valid probability distribution.实际的概率分布
    # logits: Unscaled log probabilities.假设的概率分布
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # 要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.5的学习速率最小化交叉熵
    # 梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 通过InteractiveSession类，能让我在运行图的时候，插入一些计算图
    sess = tf.InteractiveSession()

    # Train(随机梯度下降训练)
    # 1 让模型循环训练1000次,
    # 2 循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
    # 优点：既可以减少计算开销，又可以最大化地学习到数据集的总体特性
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model（评估模型）
    # 1 找出那些预测正确的标签；
    # 2 为了确定正确预测项的比例，把布尔值转换成浮点数，然后取平均值；
    # 3 计算所学习到的模型在测试数据集上面的正确率；
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


# 0.9074

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/PyCharm/tensorflow/mnist/MNIST_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    print(unparsed)
    print([sys.argv[0]] + unparsed)
    print(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
