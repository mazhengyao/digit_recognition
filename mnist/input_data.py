# D:\Anaconda\envs\tensor2\python.exe python3
# -*- coding: utf-8 -*-
# @Time : 2020/2/9
# @Author : Damon Ma
# @Email : ma_zhengyao@163.com
# @File : input_data.py
# @Software: PyCharm
# MNIST官网
# http://yann.lecun.com/exdb/mnist/

# 导入MNIST数据集，下载到MNIST_data文件夹
# train-images-idx3-ubyte.gz: 训练集，55000张训练集，5000张验证集，9,681 kb
# train-labels-idx1-ubyte.gz: 训练集标签，训练集图片对应的标签，29 kb
# t10k-images-idx3-ubyte.gz：测试集，10000张测试集，1,611 kb
# t10k-labels-idx1-ubyte.gz：测试集标签，测试集图片对应的标签，5 kb
# 这样的切分很重要，在机器学习模型设计时必须有一个单独的测试数据集不用于训练
# 而是用来评估这个模型的性能，从而更加容易把设计的模型推广到其他数据集上（泛化）


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets