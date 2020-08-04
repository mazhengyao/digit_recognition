# D:\Anaconda\envs\tensor2\python.exe python3
# -*- coding: utf-8 -*-
# @Time : 2020/2/14
# @Author : Damon Ma
# @Email : ma_zhengyao@163.com
# @File : regression.py
# @Software: PyCharm

import os

import input_data
import model
import tensorflow as tf

# 线性回归模型在MNIST上只有91%正确率！！  0.9223

# 在MNIST训练数据集中，数据是一个形状为 [60000, 784] 的张量
# 第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。
# 在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于0和1之间。
# 相对应的MNIST数据集的标签是介于0到9的数字，用来描述给定图片里表示的数字。
# one-hot vectors：一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。
# label数据是一个 [60000, 10] 的数字矩阵。
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标
# 一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。
# 交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段。
# 交叉熵是用来衡量我们的预测用于描述真相的低效性

# create model 命名regression
# tf.variable_scope()
# 在模型中开辟各自的空间，而其中的变量均在这个空间内进行管理
# 创建/调用变量的函数tf.variable() 和tf.get_variable()搭配使用
with tf.variable_scope("regression"):
    # x:待用户输入，placeholder占位符
    # placeholder的第一个参数是类型，第二个是张量
    # 输入为None:任意数量的，784像素的图像
    # 其中的784是和model中对应
    x = tf.placeholder(tf.float32, [None, 784])
    # y: 预测的概率分布[10]，归一化操作，概率之和加起来等于1
    # variables [W,b]
    # W: 权重，[784，10]，生成10个大小为784的特征图
    # b: 偏置，[10]，附加到特征图之上
    y, variables = model.regression(x)

# train训练
# 为了计算交叉熵，首先需要添加一个新的占位符用于输入正确值，MNIST中的正确标签
y_ = tf.placeholder("float", [None, 10])

# 计算交叉熵
# 首先，用 tf.log 计算 y 的每个元素的对数。
# 接下来，把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘。
# 最后，用 tf.reduce_sum 计算张量的所有元素的总和。reduce_sum()中就是按照求和的方式对矩阵降维。
# （注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。
#  对于100个数据点的预测表现比单一数据点的表现能更好地描述模型的性能。）
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练步骤
# TensorFlow拥有一张描述你各个计算单元的图，它可以自动地使用(backpropagation algorithm)
# 反向传播算法来有效地确定你的变量是如何影响你想要最小化的那个成本值的。
# 然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。

# TensorFlow用梯度下降算法（gradient descent algorithm）
# 以0.01的学习速率最小化交叉熵。
# 梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。
# TensorFlow也提供了其他许多优化算法：只要简单地调整一行代码就可以使用其他的算法。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 评估模型
# 首先找出那些预测正确的标签。
# tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
# 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签
# 比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签
# 可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 上述代码会给我们一组布尔值。
# 为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
# 例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
# 准确率 转换格式
# tf.cast()数据类型转换，布尔值转换成浮点数
# reduce_mean()就是按照某个维度求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 参数进行保存
saver = tf.train.Saver(variables)

# 开始训练
# 在一个Session里面启动模型，并且初始化变量
with tf.Session() as sess:
    # 把全部的参数放进来，进行全局初始化
    # 现在，已经设置好了模型。在运行计算之前，需要添加一个操作来初始化创建的变量
    sess.run(tf.global_variables_initializer())

    # 断点续训
    ckpt = tf.train.get_checkpoint_state(
        os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # 训练20000次
    for _ in range(20000):
        # 使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 更确切的说是随机梯度下降训练。
        # 每一个MNIST数据单元有两部分组成
        # 一张包含手写数字的图片和一个对应的标签。把这些图片设为“xs”，把这些标签设为“ys”
        # 随机抓取训练数据中的100个批处理数据点
        # 然后用这些数据点作为参数替换之前的占位符来运行train_step
        batch_xs, batch_ys = data.train.next_batch(100)
        # feed_dict：喂参数，x放的batch_xs,y_放的batch_ys
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    # 打印测试集的图像和标签
    # 所学习到的模型在测试数据集上面的正确率
    print((sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels})))
    path = saver.save(
        # 把数据存进去，把这个模型的名字存成regression.ckpt，在这里注意data文件夹的创建
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False,
        write_state=False       # 写到图中
    )    # 把数据或者说是参数或者说是模型存起来


    # 把保存模型的路径打印出来
    print('Saved:', path)
