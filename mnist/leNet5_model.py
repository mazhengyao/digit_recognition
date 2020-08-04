# D:\Anaconda\envs\tensor2\python.exe python3
# -*- coding: utf-8 -*-
# @Time : 2020/3/21
# @Author : Damon Ma
# @Email : ma_zhengyao@163.com
# @File : leNet_model.py
# @Software: PyCharm

import tensorflow as tf


# leNet5模型
def leNet5(x, keep_prob):
    """
    利用卷积神经网络模型
    :param:
        x: 数据输入，[None,784]
            None代表任意数量，784 = 28 * 28 为Mnist的图片大小，像素点
        keep_prob：保留神经元的百分比
            也可以理解为代表一个神经元的输出在dropout中保持不变的概率
    :return:
        y: 预测的概率分布 [10]，归一化操作，概率之和加起来等于1
        W_conv1: 权重，卷积核大小[5, 5, 1, 6]，长宽为5通道为1，生成6个特征图
        b_conv1: 偏置，[6]，附加到特征图之上
        W_conv2: 权重，卷积核大小[5, 5, 6, 16]，长宽为5通道为6，生成16个特征图
        b_conv2: 偏置，[16]，附加到特征图之上
        W_fc1：权重，卷积核大小[5 * 5 * 16, 120]，生成长为120的一维变量
        b_fc1：偏置，[120]，附加到特征图之上
        W_fc2：权重，卷积核大小[120, 84]，生成长为84的一维变量
        b_fc2：偏置，[84]，附加到特征图之上
        W_fc3：权重，卷积核大小[84, 10]，生成长为10的一维变量
        b_fc3：偏置，[10]，附加到特征图之上
    """

    # 规则化可以帮助防止过度配合，提高模型的适用性。
    # （让模型无法完美匹配所有的训练项。）（使用规则来使用尽量少的变量去拟合数据）
    # 规则化就是说给需要训练的目标函数加上一些规则（限制），让他们不要自我膨胀。
    # 规则化函数tf.contrib.layers.l2_regularizer
    regularizer = tf.contrib.layers.l2_regularizer(0.001)


    # 输入图片大小W×W
    # Filter大小F×F
    # 步长S
    # padding的像素数P
    # N = (W − F + 2P) / S + 1
    # 输出图片大小为N×N

    # 定义一个2*2的卷积核
    def conv2d(x, W):
        """
        定义卷积核
        :param:
            x: 数据输入，[None,784]
                None代表任意数量，784 = 28 * 28 为Mnist的图片大小，像素点
            W: 权重，[784，10]，生成10个大小为784的特征图
        :return:
            tf.nn.conv2d: 返回一个卷积核
        """

        # TensorFlow中使用tf.nn.conv2d实现卷积操作
        # input: 输入的要做卷积的图片，要求为一个张量   x
        # filter： 卷积核，要求也是一个张量  W
        # strides： 卷积时在图像每一维的步长，这是一个一维的向量，[1, strides, strides, 1]，第一位和最后一位固定必须是1
        # padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界
        # "SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
        # use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true

        # 卷积使用1步长（stride size），0边距（padding size）的模板
        # 保证输出和输入是同一个大小
        # 卷积层用来提取图像的低层特征

        # 此时卷积的输出是等于输入尺寸的
        return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')

    # 定义一个2*2的池化层
    def max_pool_2x2(x):
        """
        最大值池化操作
        :param:
            x: 数据输入，[None,784]
                None代表任意数量，784 = 28 * 28 为Mnist的图片大小，像素点
        :return:
            tf.nn.max_pool: 返回一个最大值池化操作后的tensor
        """

        # 池化输出大小 = [（输入大小 - 卷积核（过滤器）大小）／步长]+1

        # tf.nn.max_pool
        # 最大值池化操作
        # value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map
        # 依然是[batch, height, width, channels]这样的shape
        # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]
        # 因为我们不想在batch和channels上做池化，所以这两个维度设为了1
        # strides：和卷积类似，窗口在每一个维度上滑动的步长
        # 一般也是[1, stride, stride, 1]
        # padding：和卷积类似，可以取'VALID'或者'SAME'
        # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 定义一个权重变量
    def weight_variable(shape):
        """
        定义一个权重变量
        :param:
            shape: 生成张量的维度
        :return:
            tf.Variable：返回一个权重变量
        """
        # tf.truncated_normal
        # 截断的产生正态分布的随机数
        # 即随机数与均值的差值若大于两倍的标准差，则重新生成
        # shape，生成张量的维度
        # stddev，标准差
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 定义一个偏置项变量
    def bias_variable(shape):
        """
        定义一个偏置项变量
        :param:
            shape: 生成张量的维度
        :return:
            tf.Variable：返回一个偏置项变量
        """
        # tf.constant
        # 创建常量的函数原型
        # value参数为该常数的值
        # shape参数表示张量的“形状”，即维数以及每一维的大小。
        # 如果指定了shape参数，当第一个参数value是数字时，张量的所有元素都会用该数字填充
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

# 第一层：卷积层
    # 过滤器（卷积核）的尺寸为5×5，深度为6,不使用全0补充padding='VALID'，步长为1。
    # 尺寸变化：28×28×1->28×28×6

    # 定义图像
    # tf.reshape(tensor, shape, name=None)
    # tensor参数为被调整维度的张量。
    # shape参数为要调整为的形状。
    # 返回一个shape形状的新tensor。
    # 注意shape里最多有一个维度的值可以填写为 - 1，表示自动计算此维度。新数组元素数量与原数组元素数量要相等
    # x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 定义权重
    # 卷积的权重张量形状是[5, 5, 1, 6]
    # weight：5 high：5 灰度图 生成6个特征图
    W_conv1 = weight_variable([5, 5, 1, 6])

    # 定义偏置项
    # 6个特征图会存在6个偏置项
    b_conv1 = bias_variable([6])

    # 定义卷积层：提取不同特征
    # relu函数 是修正线性单元函数，非线性的激活函数
    # relu函数，max(0, x)，对矩阵运算和梯度下降有好处
    # 此处第一次卷积 padding='SAME' 步长=1，公式计算可得
    # 输出 28 * 28 * 6,长宽都为28，6个特征图
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 第二层：池化层
    # 过滤器（池化窗口）的尺寸为2×2，使用全0补充padding='SAME'，步长为2。
    # 尺寸变化：28×28×6->14×14×6
    # 池化输出大小 = [（输入大小 - 卷积核（过滤器）大小）／步长]+1
    h_pool1 = max_pool_2x2(h_conv1)


# 第三层：卷积层
    # 过滤器（卷积核）的尺寸为5×5，深度为16,不使用全0补充padding='VALID'，步长为1。
    # 尺寸变化：14×14×6->10×10×16

    # 定义权重
    # 卷积的权重张量形状是[5, 5, 6, 16]
    # weight：5 high：5 输入6个特征图 生成16个特征图
    W_conv2 = weight_variable([5, 5, 6, 16])

    # 定义偏置项
    # 16个特征图会存在16个偏置项
    b_conv2 = bias_variable([16])

    # 定义卷积层：提取不同特征
    # 此处第一次卷积 padding='VALID' 步长=1，公式计算可得
    # 输出 10 * 10 * 16,长宽都为10，16个特征图
    conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID')

    # 激活
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))

# 第四层：池化层
    # 过滤器（池化窗口）的尺寸为2×2，使用全0补充padding='SAME'，步长为2。
    # 尺寸变化：10×10×16->5×5×16
    h_pool2 = max_pool_2x2(h_conv2)


    # 将第四层池化层的输出转化为第五层全连接层的输入格式。
    # 第四层的输出为5×5×16的矩阵，然而第五层全连接层需要的输入格式为向量
    # 所以我们需要把代表每张图片的尺寸为5×5×16的矩阵拉直成一个长度为5×5×16的向量。
    # 举例说，每次训练64张图片，那么第四层池化层的输出的size为(64,5,5,16)
    # 拉直为向量，nodes=5×5×16=400,尺寸size变为(64,400)
    pool_shape = h_pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(h_pool2, [-1, nodes])


# 第五层：全连接层
    # nodes=5×5×16=400，400->120的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×400->64×120
    # 训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题。
    # 这和模型越简单越不容易过拟合，正则化限制权重的大小，使得模型不能任意拟合训练数据中的随机噪声，避免过拟合思想一致。
    # 汇总特征和信息，将信息输出，将数据拉平成一维的向量，每一层的都是和上一层全部的神经元相连的，每个像素点都有自己的权值

    # 定义权重
    # 卷积的权重张量形状是[400，120]
    W_fc1 = weight_variable([nodes, 120])

    # 参数regularizer不为空，启动正则化
    if regularizer != None:
        # tf.add_to_collection是把多个变量放入一个自己用引号命名的集合里，也就是把多个变量统一放在一个列表中。
        # 在深度学习中，通常用这函数存放不同层中的权值和偏置参数
        # 也就是把所有可学习参数利用tf.contrib.layers.l2_regularizer(regular_num)(w)
        # 得到norm后，都放到’losses’的列表中作为正则项
        # 然后使用tf.add_n函数将他们和原本的loss相加，得到含有正则的loss
        tf.add_to_collection('losses', regularizer(W_fc1))

    # 定义偏置项
    # 120个特征图会存在120个偏置项
    b_fc1 = bias_variable([120])

    # 全连接层的卷积操作：直接相乘
    # matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘
    # 输出 120 一维变量
    h_fc1 = tf.nn.relu(tf.matmul(reshaped, W_fc1) + b_fc1)


    # dropout 操作，可以扔掉一些值， 减少过拟合
    # 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
    # 这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。
    # TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
    # 所以用dropout的时候可以不用考虑scale。

    # 底层其实就是降低上一层某些输入的权重scale，甚至置为0
    # 升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
    # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5
    # 根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0

    # keep_prob：保留神经元的百分比，也可以理解为代表一个神经元的输出在dropout中保持不变的概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第六层：全连接层
    # 120->84的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×120->64×84

    # 定义权重
    # 卷积的权重张量形状是[120, 84]
    W_fc2 = weight_variable([120, 84])

    # 参数regularizer不为空，启动正则化
    if regularizer != None:
        # tf.add_to_collection是把多个变量放入一个自己用引号命名的集合里，也就是把多个变量统一放在一个列表中。
        # 在深度学习中，通常用这函数存放不同层中的权值和偏置参数
        # 也就是把所有可学习参数利用tf.contrib.layers.l2_regularizer(regular_num)(w)
        # 得到norm后，都放到’losses’的列表中作为正则项
        # 然后使用tf.add_n函数将他们和原本的loss相加，得到含有正则的loss
        tf.add_to_collection('losses', regularizer(W_fc2))

    # 定义偏置项
    # 84个特征图会存在84个偏置项
    b_fc2 = bias_variable([84])

    # 全连接层的卷积操作：直接相乘
    # matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # dropout 操作，可以扔掉一些值， 减少过拟合
    # keep_prob：保留神经元的百分比，也可以理解为代表一个神经元的输出在dropout中保持不变的概率
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 第七层：全连接层（近似表示）
    # 84->10的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×10。
    # 最后，64×10的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
    # 即得到最后的分类结果。

    # 定义权重
    # 卷积的权重张量形状是[84, 10]
    W_fc3= weight_variable([84, 10])

    # 参数regularizer不为空，启动正则化
    if regularizer != None:
        # tf.add_to_collection是把多个变量放入一个自己用引号命名的集合里，也就是把多个变量统一放在一个列表中。
        # 在深度学习中，通常用这函数存放不同层中的权值和偏置参数
        # 也就是把所有可学习参数利用tf.contrib.layers.l2_regularizer(regular_num)(w)
        # 得到norm后，都放到’losses’的列表中作为正则项
        # 然后使用tf.add_n函数将他们和原本的loss相加，得到含有正则的loss
        tf.add_to_collection('losses', regularizer(W_fc3))

    # 定义偏置项
    # 10个特征图会存在10个偏置项
    b_fc3 = bias_variable([10])

    # 最后的softmax分类 得到预测结果的概率为1*1*10
    # softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
    y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    # 返回预测概率
    # 返回参数用于保存模型
    return y, [W_conv1, b_conv1,
               W_conv2, b_conv2,
               W_fc1, b_fc1,
               W_fc2, b_fc2,
               W_fc3, b_fc3]