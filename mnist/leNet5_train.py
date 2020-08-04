# D:\Anaconda\envs\tensor2\python.exe python3
# -*- coding: utf-8 -*-
# @Time : 2020/3/26
# @Author : Damon Ma
# @Email : ma_zhengyao@163.com
# @File : leNet5_train.py
# @Software: PyCharm

# TensorFlow也是在Python外部完成其主要工作，但是进行了改进以避免这种开销。
# 其并没有采用在Python外部独立运行某个耗时操作的方式，而是先让我们描述一个交互操作图，然后完全将其运行在Python外部。
import os
import model
import tensorflow as tf
import input_data
from mnist import leNet5_model

# mnist是一个轻量级的类。
# 它以Numpy数组的形式存储着训练、校验和测试数据集。
# 同时提供了一个函数，用于在迭代中获得minibatch。
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# 在最终测试集上的准确率大概是99.2%！！

# 卷积lenet5 命名leNet5
# tf.variable_scope()
# 在模型中开辟各自的空间，而其中的变量均在这个空间内进行管理
# 创建/调用变量的函数tf.variable() 和tf.get_variable()搭配使用
with tf.variable_scope('leNet5'):
    # x:待用户输入，placeholder占位符
    # placeholder的第一个参数是类型，第二个是张量
    # 输入为None:任意数量的，784像素的图像
    # 其中的784是和model中对应
    x = tf.placeholder(tf.float32, [None, 784], name='x')

    # keep_prob：保留神经元的百分比
    keep_prob = tf.placeholder(tf.float32)

    # y: 预测的概率分布[10]，归一化操作，概率之和加起来等于1
    # variables [W,b]
    # W: 权重，[784，10]，生成10个大小为784的特征图
    # b: 偏置，[10]，附加到特征图之上
    y, variables = leNet5_model.leNet5(x, keep_prob)

# train训练
# 为了计算交叉熵，首先需要添加一个新的占位符用于输入正确值，MNIST中的正确标签
y_ = tf.placeholder(tf.float32, [None, 10], name='y')

# 计算交叉熵
# 首先，用 tf.log 计算 y 的每个元素的对数。
# 接下来，把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘。
# 最后，用 tf.reduce_sum 计算张量的所有元素的总和。reduce_sum()中就是按照求和的方式对矩阵降维。
# （注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。
#  对于100个数据点的预测表现比单一数据点的表现能更好地描述模型的性能。）
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 平均交叉熵
cross_entropy_mean = tf.reduce_mean(cross_entropy)
# 损失函数
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

# TensorFlow拥有一张描述你各个计算单元的图，它可以自动地使用(backpropagation algorithm)
# 反向传播算法来有效地确定你的变量是如何影响你想要最小化的那个成本值的。
# 然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。
# AdamOptimizer是TensorFlow中实现Adam算法的优化器。
# Adam即Adaptive Moment Estimation（自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正。
# Adam 算法相对于其它种类算法有一定的优越性，是比较常用的算法之一。 1×10^(-4)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

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

# 更加复杂的ADAM优化器来做梯度最速下降
# 在feed_dict中加入额外的参数keep_prob来控制dropout比例
# 然后每100次迭代输出一次日志
# 在最终测试集上的准确率大概是99.2%
with tf.Session() as sess:

    # merge_all可以将所有summary全部保存到磁盘，以便tensorboard显示。
    # 如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
    # 合并参数，操作符
    merged_summary_op = tf.summary.merge_all()

    # 指定一个文件用来保存图。
    # 将参数的路径、输入输出图放到哪里
    summary_writer = tf.summary.FileWriter('./graph', sess.graph)
    # 把图加进来
    summary_writer.add_graph(sess.graph)

    # 把全部的参数放进来，进行全局初始化
    # 现在，已经设置好了模型。在运行计算之前，需要添加一个操作来初始化创建的变量
    sess.run(tf.global_variables_initializer())

    # 对于这样的卷积训练一般要做10000-20000次的循环
    for i in range(20000):
        # 使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 更确切的说是随机梯度下降训练。
        # 随机抓取训练数据中的50个批处理数据点
        # 然后用这些数据点作为参数替换之前的占位符来运行train_step
        # 定义batch的大小 训练集
        batch = data.train.next_batch(100)
        # 每隔100次准确率做一次打印
        if i % 100 == 0:
            # batch[0] 图像
            # batch[1] 标签
            # eval()其实就是tf.Tensor的Session.run()的另外一种写法
            # sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})
            # keep_prob: 1.0
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
            # 每一百次打印次数和准确率
            print('step %d, training accuracy %g' % (i, train_accuracy))
            # 保存ckpt模型文件
            path = saver.save(
                sess, os.path.join(os.path.dirname(__file__), 'data', 'leNet5.ckpt'),
                write_meta_graph=False, write_state=False
            )
            # 每一百次打印测试集的成功率 观察变换
            print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))
        # 整个模型的训练可以通过反复地运行train_step
        # keep_prob: 0.5 丢弃50%的神经元
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # 训练完成后，打印测试集的准确率
    # keep_prob: 1.0 保留全部神经元
    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

    print("#########################最终保存#########################")

    # 保存ckpt模型文件
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'leNet5.ckpt'),
        write_meta_graph=False, write_state=False
    )

    # 打印保存路径
    print('Saved:', path)