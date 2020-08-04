# D:\Anaconda\envs\tensor2\python.exe python3
# -*- coding: utf-8 -*-
# @Time : 2020/3/31
# @Author : Damon Ma
# @Email : ma_zhengyao@163.com
# @File : main.py
# @Software: PyCharm


import json
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from mnist import model
from mnist import leNet5_model
from PIL import Image
from numpy import uint8


# x:待用户输入，placeholder占位符
# placeholder的第一个参数是类型，第二个是张量
# 输入为None:任意数量的，784像素的图像
# 其中的784是和model中对应
x = tf.placeholder('float', [None, 784])

# 定义Session
sess = tf.Session()

# tf.variable_scope()
# 在模型中开辟各自的空间，而其中的变量均在这个空间内进行管理
# 创建/调用变量的函数tf.variable() 和tf.get_variable()搭配使用

# 从model中拿出线性回归模型
with tf.variable_scope('regression'):
    # 调用regression函数得到模型
    y1, variables = model.regression(x)

# 参数进行保存
saver = tf.train.Saver(variables)

# restore的用法是将训练好的参数提取出来
# 根据路径提出线性回归模型的相关参数
saver.restore(sess, 'mnist/data/regression.ckpt')

# keep_prob：保留神经元的百分比
keep_prob = tf.placeholder('float')

# 从model中拿出卷积神经网络模型
with tf.variable_scope('convolutional'):
    # keep_prob = tf.placeholder('float')
    # 调用convolutional函数得到模型
    y2, variables2 = model.convolutional(x, keep_prob)

# 参数进行保存
saver = tf.train.Saver(variables2)

# restore的用法是将训练好的参数提取出来
# 根据路径提出卷积神经网络模型的相关参数
saver.restore(sess, 'mnist/data/convolutional.ckpt')


# 从leNet5_model中拿出leNet-5神经网络模型
with tf.variable_scope('leNet5'):
    # keep_prob = tf.placeholder('float')
    # 调用leNet5函数得到模型
    y3, variables3 = leNet5_model.leNet5(x, keep_prob)

# 参数进行保存
saver = tf.train.Saver(variables3)

# restore的用法是将训练好的参数提取出来
# 根据路径提出leNet-5神经网络模型的相关参数
saver.restore(sess, 'mnist/data/leNet5.ckpt')

# 线性回归模型将界面上输入 转换成list
def regression(input):
    """
      转换成list
      Y = W * x + b
      拟合线性回归
      :param:
          input: 数据输入
      :return:
          list: 转换后的List类型数据
    """

    # y1：也就是执行regression函数得到y1
    # feed_dict: 喂参数x
    # flatten()返回一个一维数组
    # tolist()将数据变成List类型
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


# 卷积神经网络模型将界面上输入 转换成list
def convolutional(input):
    """
      转换成list
      Y = W * x + b
      拟合线性回归
      :param:
          input: 数据输入
      :return:
          list: 转换后的List类型数据
    """

    # keep_prob：保留神经元的百分比
    # y2：也就是执行convolutional函数得到y2
    # feed_dict: 喂参数x
    # flatten()返回一个一维数组
    # tolist()将数据变成List类型
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

# lenet5神经网络模型将界面上输入 转换成list
def lenet5(input):
    """
      转换成list
      Y = W * x + b
      拟合线性回归
      :param:
          input: 数据输入
      :return:
          list: 转换后的List类型数据
    """

    # keep_prob：保留神经元的百分比
    # y3：也就是执行leNet5函数得到y3
    # feed_dict: 喂参数x
    # flatten()返回一个一维数组
    # tolist()将数据变成List类型
    return sess.run(y3, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


# 以上步骤之后，三个模型的输入、如何把数据喂进来以及如何跑这个模型已经完成

# 用Flask做接口，前后端传递数据

# 第一部分，初始化：所有的Flask都必须创建程序实例，
# web服务器使用wsgi协议，把客户端所有的请求都转发给这个程序实例
# 程序实例是Flask的对象，一般情况下用如下方法实例化
# Flask类只有一个必须指定的参数，即程序主模块或者包的名字，__name__是系统变量，该变量指的是本py文件的文件名

app = Flask(__name__)

#  第二部分，路由和视图函数：
#  客户端发送url给web服务器，web服务器将url转发给flask程序实例，程序实例
#  需要知道对于每一个url请求启动那一部分代码，所以保存了一个url和python函数的映射关系。
#  处理url和函数之间关系的程序，称为路由
#  在flask中，定义路由最简便的方式，是使用程序实例的app.route装饰器，把装饰的函数注册为路由

# 定义一个注解，路由，表示前端传进来之后应该用哪个接口
@app.route('/api/mnist', methods=['post'])
def mnist():
    """
      识别服务路由
      :param:
      :return:
          jsonify: 处理后的json数据
    """

    # 数据都是0-255的数字，归一化为0-1之间的数
    # 做一个数组的换算，模型定义的形状就是1*784的形状
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)

    # 经过线性回归模型的预测得到的识别结果
    output1 = regression(input)
    # 经过卷积神经网络模型的预测得到的识别结果
    output2 = convolutional(input)
    # 经过lenet-5神经网络模型的预测得到的识别结果
    output3 = lenet5(input)
    # 将结果结果封装成json传给前端
    return jsonify(results=[output1, output2, output3])


# 测试接口
@app.route('/test')
def test():
    """
      测试接口
      :param:
      :return:
          string: '测试接口'
    """
    return '测试接口'


# 默认绑定 ‘/’ 访问的界面
@app.route('/')
def main():
    """
      将canvas.html和main.py绑定
      :param:
      :return:
          render_template: html模板内容
    """
    #  创建视图函数，将canvas.html模板内容进行渲染返回
    return render_template('canvas.html')


# 绑定 ‘/main’ 访问的界面
@app.route('/main')
def mainCanvas():
    """
      绑定 ‘/main’ 访问的界面
      :param:
      :return:
          render_template: html模板内容
    """
    #  创建视图函数，将canvas.html模板内容进行渲染返回
    return render_template('canvas.html')


# 绑定 ‘/welcome’ 访问的界面
@app.route('/welcome')
def welcome():
    """
      绑定 ‘/welcome’ 访问的界面
      :param:
      :return:
          render_template: html模板内容
    """
    #  创建视图函数，将welcome.html模板内容进行渲染返回
    return render_template('welcome.html')


# 绑定 ‘/about’ 访问的界面
@app.route('/about')
def about():
    """
      绑定 ‘/about’ 访问的界面
      :param:
      :return:
          render_template: html模板内容
    """
    #  创建视图函数，将about.html模板内容进行渲染返回
    return render_template('about.html')


#  第三部分：程序实例用run方法启动flask集成的开发web服务器
#  __name__ == '__main__'是python常用的方法，表示只有直接启动本脚本时候，才用app.run方法
#  如果是其他脚本调用本脚本，程序假定父级脚本会启用不同的服务器，因此不用执行app.run()
#  服务器启动后，会启动轮询，等待并处理请求。轮询会一直请求，直到程序停止。

if __name__ == '__main__':
    # 可以开启debug模式
    app.debug = False
    # 127.0.0.1/8000
    app.run(host='127.0.0.1', port=8000)

    # 浏览器将请求给web服务器，web服务器将请求给app,
    # app收到请求，通过路由找到对应的视图函数，然后将请求处理，得到一个响应response
    # 然后app将响应返回给web服务器，
    # web服务器返回给浏览器，
    # 浏览器展示给用户观看，流程完毕。