# D:\Anaconda\envs\tensor2\python.exe python3
# -*- coding: utf-8 -*-
# @Time : 2020/2/9
# @Author : Damon Ma
# @Email : ma_zhengyao@163.com
# @File : model.py
# @Software: PyCharm

import tensorflow as tf

# Y=W*x+b
# 拟合线性回归:输入参数 给出判断 数学问题
# 利用tensorflow构造公式，相当于只有一层
def regression(x):
    """
    利用线性回归模型
    Y = W * x + b
    拟合线性回归
    :param:
        x: 数据输入，[None,784]
            None代表任意数量，784 = 28 * 28 为Mnist的图片大小，像素点
    :return:
        y: 预测的概率分布 [10]，归一化操作，概率之和加起来等于1
        W: 权重，[784，10]，生成10个大小为784的特征图
        b: 偏置，[10]，附加到特征图之上
    """
    # tf.Variable(initializer,name)
    # 参数initializer是初始化参数，name是可自定义的变量名称
    # 一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中
    # 它们可以用于计算输入值，也可以在计算中被修改
    # tf.zeros()
    # 用全为零的张量来初始化W和b
    # 因为我们要学习W和b的值，它们的初值可以随意设置
    # tf.matmul（a,b）
    # 将矩阵a乘以矩阵b，生成a * b

    # W为权重：weights
    # W的维度是[784，10]，784 * 10的二维数组
    # 因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量
    # 每一位对应不同数字类，产生10个特征图  10
    W = tf.Variable(tf.zeros([784, 10]), name="W")

    # b为偏置：biases
    # b的维度是[10]，可以直接把它加到输出上面，也就是10个特征图上
    # 一维数组里面放10个值
    b = tf.Variable(tf.zeros([10]), name="b")

    # 模型实现！
    # 输入到tf.nn.softmax函数里面,softmax做简单的线性运算
    # 利用softmax分类器，把计算出的结果分为一个10维向量来表示概率分布
    # 每个向量分别依次代表0-9的数字，并且概率之和加起来等于1，完成01归一化操作
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    return y, [W, b]


# 卷积模型:多层卷积
def convolutional(x, keep_prob):
    """
    利用卷积神经网络模型
    :param:
        x: 数据输入，[None,784]
            None代表任意数量，784 = 28 * 28 为Mnist的图片大小，像素点
        keep_prob：保留神经元的百分比
            也可以理解为代表一个神经元的输出在dropout中保持不变的概率
    :return:
        y: 预测的概率分布 [10]，归一化操作，概率之和加起来等于1
        W_conv1: 权重，卷积核大小[5, 5, 1, 32]，长宽为5通道为1
            生成32个大小为长宽为28的特征图
        b_conv1: 偏置，[32]，附加到特征图之上
        W_conv2: 权重，卷积核大小[5, 5, 32, 64]，长宽为5通道为32
            生成64个大小为长宽为14的特征图
        b_conv2: 偏置，[64]，附加到特征图之上
        W_fc1：权重，卷积核大小[7 * 7 * 64, 1024]，输入一维向量7 * 7 * 64
            生成1个长为1024的一维变量
        b_fc1：偏置，[1024]，附加到特征图之上
        W_fc2：权重，卷积核大小[1024, 10]，输入一维向量1024
            生成1个长为10的一维变量
        b_fc2：偏置，[10]，附加到特征图之上
    """

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

# 第一层：卷积层 28*28*1 --> 28*28*32
    # 由一个卷积核接一个max pooling完成。
    # 卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]。
    # 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
    # 而对于每一个输出通道都有一个对应的偏置量。

    # 定义图像
    # tf.reshape(tensor, shape, name=None)
    # tensor参数为被调整维度的张量。
    # shape参数为要调整为的形状。
    # 返回一个shape形状的新tensor。
    # 注意shape里最多有一个维度的值可以填写为 - 1，表示自动计算此维度。
    # x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 定义权重
    # 卷积的权重张量形状是[5, 5, 1, 32]
    # weight：5 high：5 灰度图 生成32个特征图
    W_conv1 = weight_variable([5, 5, 1, 32])

    # 定义偏置项
    # 32个特征图会存在32个偏置项
    b_conv1 = bias_variable([32])

    # 定义卷积层：提取不同特征
    # relu函数 是修正线性单元函数，非线性的激活函数
    # relu函数，max(0, x)，对矩阵运算和梯度下降有好处
    # 此处第一次卷积 padding='SAME' 步长=1，公式计算可得
    # 输出 28 * 28 * 32,长宽都为28，32个特征图
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 第二层：池化层（下采样） 28*28*32 --> 14*14*32
    # 定义池化层：保证特征不变，将数据压缩
    # max pooling,最大池化，将池化窗口数据取最大值来代表整个池化窗口
    # 池化输出大小 = [（输入大小 - 卷积核（过滤器）大小）／步长]+1 ： 28-2/2 +1
    # 输出 14 * 14 * 32,长宽都为14，32个特征图
    h_pool1 = max_pool_2x2(h_conv1)


# 第三层：卷积层  14*14*32 --> 14*14*64
    # 为了构建一个更深的网络，会把几个类似的层堆叠起来。
    # 第二层中，每个5x5的patch会得到64个特征。

    # 定义权重
    # 卷积的权重张量形状是 [5, 5, 32, 64]
    # weight：5 high：5 输入32个特征图 生成64个特征图
    W_conv2 = weight_variable([5, 5, 32, 64])

    # 定义偏置项
    # 64个特征图会存在64个偏置项
    b_conv2 = bias_variable([64])

    # 定义卷积层：提取不同特征
    # 此处第一次卷积 padding='SAME' 步长=1，公式计算可得
    # 输出 14 * 14 * 64,长宽都为14，64个特征图
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第四层：池化层（下采样） 14*14*64 --> 7*7*64
    # 定义池化层：保证特征不变，将数据压缩
    # max pooling,最大池化，将池化窗口数据取最大值来代表整个池化窗口
    # 池化输出大小 = [（输入大小 - 卷积核（过滤器）大小）／步长]+1 ： 14-2/2 +1
    # 输出 7 * 7 * 64,长宽都为14，64个特征图
    h_pool2 = max_pool_2x2(h_conv2)

# 第五层：全连接层 7*7*64 --> 1024
    # 全连接层：full connection
    # 汇总特征和信息，将信息输出，将数据拉平成一维的向量
    # 每一层的都是和上一层全部的神经元相连的，每个像素点都有自己的权值

    # 定义权重
    # 权重张量形状是[7 * 7 * 64, 1024]
    # 输入为7 * 7 * 64，输出为1024 一维向量
    # 二维张量，第一个参数7*7*64的patch，也可以认为是只有一行7*7*64个数据的卷积
    # 第二个参数代表卷积个数共1024个
    W_fc1 = weight_variable([7 * 7 * 64, 1024])

    # 定义偏置项
    # 1024个特征图会存在1024个偏置项
    b_fc1 = bias_variable([1024])

    # 全连接层的池化
    # 将第二层卷积池化结果reshape成只有一行7*7*64个数据
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # 全连接层的卷积操作：直接相乘
    # 卷积操作，结果是1*1*1024，单行乘以单列等于1*1矩阵
    # matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘
    # 自动认为是前行向量后列向量
    # 输出 1024 一维变量
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


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

# 第六层：全连接层 1024 --> 10
    # 定义权重
    # 权重张量形状是[1024, 10]
    # 二维张量，1*1024矩阵卷积，共10个卷积，对应我们开始的ys长度为10
    # 输入为1024，输出为10一维向量
    W_fc2 = weight_variable([1024, 10])

    # 定义偏置项
    # 10个特征图会存在10个偏置项
    b_fc2 = bias_variable([10])

    # 最后的softmax分类
    # 结果为1*1*10
    # softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]