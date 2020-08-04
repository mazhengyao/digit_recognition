# D:\Anaconda\envs\tensor2\python.exe python3
# -*- coding: utf-8 -*-
# @Time : 2020/3/14
# @Author : Damon Ma
# @Email : ma_zhengyao@163.com
# @File : leNet.py
# @Software: PyCharm

######leNet5理论实现######

from skimage import io, transform
import os
import glob
import numpy as np
import tensorflow as tf

# 将所有的MNIST数据集中的图片28*28*1重新设置尺寸为32*32*1
# w宽度，h高度，c通道数
w = 32
h = 32
c = 1

#模型的保存路径
MODEL_SAVE_PATH = "./data/"
#模型保存的文件名
MODEL_NAME = "leNet5_re.ckpt"

# mnist数据集中训练数据和测试数据保存地址
# train_path = "./MNIST/train/"
# test_path = "./MNIST/test/"
train_path = "D:\\graduationProject\\tensorMnist\\mnist\\MNIST\\train\\"
test_path = "D:\\graduationProject\\tensorMnist\\mnist\\MNIST\\test\\"


# 读取图片及其标签函数
def read_image(path):
    """
    读取图片及其标签函数
    :param:
        path: 存放路径
    :return:
        np.asarray: 重置后的asarray数据
    """
    # 读取路径下的 子文件目录
    label_dir = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    # 用以存放图像
    images = []
    # 用以存放标签
    labels = []
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
    # 同时列出数据和数据下标，一般用在 for 循环当中
    for index, folder in enumerate(
            label_dir):
        # 获取指定目录下的所有图片
        for img in glob.glob(folder + '/*.png'):
            print("reading the image:%s" % img)
            # 读取图片文件
            image = io.imread(img)
            # 图片重置尺寸
            image = transform.resize(image, (w, h, c))
            # 存储到列表中
            images.append(image)
            labels.append(index)

    # array和asarray都可以将结构数据转化为ndarray
    # 但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会
    return np.asarray(images, dtype=np.float32), np.asarray(labels,
                                                            dtype=np.int32)

# 读取训练数据
train_data, train_label = read_image(train_path)
# 读取测试数据
test_data, test_label = read_image(test_path)

# 打乱训练数据
train_image_num = len(train_data)
# arange(start，stop, step, dtype=None)
# 根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。
train_image_index = np.arange(
    train_image_num)
# 乱序函数，多维时只对一维乱序
np.random.shuffle(train_image_index)
# 乱序后的图像数据
train_data = train_data[train_image_index]
# 乱序后的标签数据
train_label = train_label[train_image_index]

# 打乱测试数据
test_image_num = len(test_data)
# arange(start，stop, step, dtype=None)
# 根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。
test_image_index = np.arange(test_image_num)
# 乱序函数，多维时只对一维乱序
np.random.shuffle(test_image_index)
# 乱序后的图像数据
test_data = test_data[test_image_index]
# 乱序后的标签数据
test_label = test_label[test_image_index]

# 搭建leNet-5
# x:placeholder 输入
# y_:placeholder 正确的标签
x = tf.placeholder(tf.float32, [None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, [None], name='y_')


def inference(input_tensor, train, regularizer):
    """
       利用leNet-5卷积神经网络模型
       :param:
           input_tensor: 数据输入的tensor
           train：是否为训练集
           regularizer：正则化
       :return:
           logit: 预测的概率分布 [10]，归一化操作，概率之和加起来等于1
           conv1_weights: 权重，第一层卷积
           conv1_biases: 偏置，第一层卷积
           conv2_weights: 权重，第二层卷积
           conv2_biases: 偏置，第二层卷积
           fc1_weights：权重，第一层全连接
           fc1_biases：偏置，第一层全连接
           fc2_weights：权重，第二层全连接
           fc2_biases：偏置，第二层全连接
           fc3_weights：权重，第三层全连接
           fc3_biases：偏置，第三层全连接
       """

# 第一层：卷积层
    # 过滤器（卷积核）的尺寸为5×5，深度为6,不使用全0补充padding='VALID'，步长为1。
    # 尺寸变化：32×32×1->28×28×6
    with tf.variable_scope('layer1-conv1'):

        # 定义权重
        # 卷积的权重张量形状是[5, 5, 1, 6]
        conv1_weights = tf.get_variable('weight', [5, 5, c, 6], initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 定义偏置项
        # 6个特征图会存在6个偏置项
        conv1_biases = tf.get_variable('bias', [6], initializer=tf.constant_initializer(0.0))

        # 定义卷积层：提取不同特征
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')

        # relu函数 是修正线性单元函数，非线性的激活函数
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

# 第二层：池化层
    # 过滤器（池化窗口）的尺寸为2×2，使用全0补充padding='SAME'，步长为2。
    # 尺寸变化：28×28×6->14×14×6
    with tf.name_scope('layer2-pool1'):
        # 定义池化层：保证特征不变，将数据压缩
        # max pooling,最大池化，将池化窗口数据取最大值来代表整个池化窗口
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第三层：卷积层
    # 过滤器（卷积核）的尺寸为5×5，深度为16,不使用全0补充padding='VALID'，步长为1。
    # 尺寸变化：14×14×6->10×10×16
    with tf.variable_scope('layer3-conv2'):

        # 定义权重
        # 卷积的权重张量形状是[5, 5, 6, 16]
        conv2_weights = tf.get_variable('weight', [5, 5, 6, 16],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 定义偏置项
        # 16个特征图会存在16个偏置项
        conv2_biases = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.0))

        # 定义卷积层：提取不同特征
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')

        # relu函数 是修正线性单元函数，非线性的激活函数
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

# 第四层：池化层
    # 过滤器（池化窗口）的尺寸为2×2，使用全0补充padding='SAME'，步长为2。
    # 尺寸变化：10×10×6->5×5×16
    with tf.variable_scope('layer4-pool2'):
        # 定义池化层：保证特征不变，将数据压缩
        # max pooling,最大池化，将池化窗口数据取最大值来代表整个池化窗口
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 将第四层池化层的输出转化为第五层全连接层的输入格式。
    # 第四层的输出为5×5×16的矩阵，然而第五层全连接层需要的输入格式为向量
    # 所以我们需要把代表每张图片的尺寸为5×5×16的矩阵拉直成一个长度为5×5×16的向量。
    # 举例说，每次训练64张图片，那么第四层池化层的输出的size为(64,5,5,16)
    # 拉直为向量，nodes=5×5×16=400,尺寸size变为(64,400)
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])

# 第五层：全连接层
    # nodes=5×5×16=400，400->120的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×400->64×120
    # 训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题。
    # 这和模型越简单越不容易过拟合思想一致
    # 正则化限制权重的大小，使得模型不能任意拟合训练数据中的随机噪声，以此达到避免过拟合思想一致。
    with tf.variable_scope('layer5-fc1'):

        # 定义权重
        # 卷积的权重张量形状是[400，120]
        fc1_weights = tf.get_variable('weight', [nodes, 120],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 参数regularizer不为空，启动正则化
        if regularizer != None:
            # tf.add_to_collection是把多个变量放入一个自己用引号命名的集合里，也就是把多个变量统一放在一个列表中。
            # 在深度学习中，通常用这函数存放不同层中的权值和偏置参数
            # 也就是把所有可学习参数利用tf.contrib.layers.l2_regularizer(regular_num)(w)
            # 得到norm后，都放到’losses’的列表中作为正则项
            # 然后使用tf.add_n函数将他们和原本的loss相加，得到含有正则的loss
            tf.add_to_collection('losses', regularizer(fc1_weights))

        # 定义偏置项
        # 120个特征图会存在120个偏置项
        fc1_biases = tf.get_variable('bias', [120],
                                        initializer=tf.constant_initializer(0.1))

        # 全连接层的卷积操作：直接相乘
        # matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        # 如果是训练集
        if train:
            # dropout50%的神经元，避免过拟合
            fc1 = tf.nn.dropout(fc1, 0.5)

# 第六层：全连接层
    # 120->84的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×120->64×84
    with tf.variable_scope('layer6-fc2'):

        # 定义权重
        # 卷积的权重张量形状是[120, 84]
        fc2_weights = tf.get_variable('weight', [120, 84],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 参数regularizer不为空，启动正则化
        if regularizer != None:
            # tf.add_to_collection是把多个变量放入一个自己用引号命名的集合里，也就是把多个变量统一放在一个列表中。
            # 在深度学习中，通常用这函数存放不同层中的权值和偏置参数
            # 也就是把所有可学习参数利用tf.contrib.layers.l2_regularizer(regular_num)(w)
            # 得到norm后，都放到’losses’的列表中作为正则项
            # 然后使用tf.add_n函数将他们和原本的loss相加，得到含有正则的loss
            tf.add_to_collection('losses', regularizer(fc2_weights))

        # 定义偏置项
        # 120个特征图会存在120个偏置项
        fc2_biases = tf.get_variable('bias', [84],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 全连接层的卷积操作：直接相乘
        # matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

        # 如果是训练集
        if train:
            # dropout50%的神经元，避免过拟合
            fc2 = tf.nn.dropout(fc2, 0.5)

# 第七层：全连接层（近似表示）
    # 84->10的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×10。
    # 最后，64×10的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
    # 即得到最后的分类结果。
    with tf.variable_scope('layer7-fc3'):

        # 定义权重
        # 卷积的权重张量形状是[84, 10]
        fc3_weights = tf.get_variable('weight', [84, 10],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 参数regularizer不为空，启动正则化
        if regularizer != None:
            # tf.add_to_collection是把多个变量放入一个自己用引号命名的集合里，也就是把多个变量统一放在一个列表中。
            # 在深度学习中，通常用这函数存放不同层中的权值和偏置参数
            # 也就是把所有可学习参数利用tf.contrib.layers.l2_regularizer(regular_num)(w)
            # 得到norm后，都放到’losses’的列表中作为正则项
            # 然后使用tf.add_n函数将他们和原本的loss相加，得到含有正则的loss
            tf.add_to_collection('losses', regularizer(fc3_weights))

        # 定义偏置项
        # 10个特征图会存在10个偏置项
        fc3_biases = tf.get_variable('bias', [10],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 得到预测结果的概率
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    # 返回预测概率
    # 返回参数用于保存模型
    return logit,[conv1_weights, conv1_biases,
                  conv2_weights, conv2_biases,
                  fc1_weights, fc1_biases,
                  fc2_weights, fc2_biases,
                  fc3_weights, fc3_biases]


# 正则化
regularizer = tf.contrib.layers.l2_regularizer(0.001)
y,variables = inference(x, True, regularizer)
# 交叉熵
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
# 平均交叉熵
cross_entropy_mean = tf.reduce_mean(cross_entropy)
# 损失函数
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
# 最小化损失函数
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
# 预测和实际equal比较，tf.equal函数会得到True或False
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)

# accuracy首先将tf.equal比较得到的布尔值转为float型，即True转为1.，False转为0，最后求平均值，即一组样本的正确率。
# 比如：一组5个样本，tf.equal比较为[True False True False False],转化为float型为[1. 0 1. 0 0],准确率为2./5=40%。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 每次获取batch_size个样本进行训练或测试
def get_batch(data, label, batch_size):
    """
    获取样本进行训练或测试
    :param:
        data: 数据输入，图片
        label: 数据输入，标签
        batch_size: 样本大小
    :yield:
        data：数据集中的指定的图片
        label: 数据集中的指定的标签
    """
    for start_index in range(0, len(data) - batch_size + 1, batch_size):
        # 从张量中提取切片
        slice_index = slice(start_index, start_index + batch_size)
        # yield 返回
        # 下次从上次返回的位置继续
        yield data[slice_index], label[slice_index]

# 保存参数到ckpt文件中
# 反向传播算法进行修改
saver = tf.train.Saver(variables)

# 创建Session会话
with tf.Session() as sess:

    # 初始化所有变量(权值，偏置等)
    # 把全部的参数放进来，进行全局初始化
    # 现在，已经设置好了模型。在运行计算之前，需要添加一个操作来初始化创建的变量
    sess.run(tf.global_variables_initializer())

    # 断点续训
    # ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)

    # 将所有样本训练10次
    # 每次训练中以64个为一组训练完所有样本。
    # train_num可以设置大一些。
    train_num = 10
    batch_size = 64

    # 开始训练train_num次
    for i in range(train_num):

        # 训练损失，训练正确率，数据量 初始化
        train_loss, train_acc, batch_num = 0, 0, 0

        # 获取训练集，进行训练
        for train_data_batch, train_label_batch in get_batch(train_data, train_label, batch_size):

            # 重复执行
            # train_op：最小化损失函数
            # loss：损失函数
            # accuracy:正确率
            _, err, acc = sess.run([train_op, loss, accuracy],
                                   feed_dict={x: train_data_batch, y_: train_label_batch})

            # 样本训练时候叠加
            # 训练损失
            train_loss += err
            # 正确次数
            train_acc += acc
            # 训练的次数
            batch_num += 1

        # 打印训练集的损失
        print("train loss:", train_loss / batch_num)
        # 打印训练集的正确率
        print("train acc:", train_acc / batch_num)

        # 测试集损失，训练正确率，数据量 初始化
        test_loss, test_acc, batch_num = 0, 0, 0

        # 获取测试集，进行检测
        for test_data_batch, test_label_batch in get_batch(test_data, test_label, batch_size):

            # 重复执行
            # loss：损失函数
            # accuracy:正确率
            err, acc = sess.run([loss, accuracy],
                                feed_dict={x: test_data_batch, y_: test_label_batch})

            # 样本测试时候叠加
            # 测试损失
            test_loss += err
            # 正确次数
            test_acc += acc
            # 训练的次数
            batch_num += 1

        # 打印测试集的损失
        print("test loss:", test_loss / batch_num)
        # 打印测试集的正确率
        print("test acc:", test_acc / batch_num)


        print("#########################保存#########################")

        # 保存ckpt模型文件
        path = saver.save(
            sess, os.path.join(os.path.dirname(__file__), MODEL_SAVE_PATH, MODEL_NAME),
            write_meta_graph=False, write_state=False
        )

        # 打印保存路径
        print('Saved:', path)
