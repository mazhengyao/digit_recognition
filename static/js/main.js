/**
 * @Description: 识别功能主逻辑
 * @author Damon Ma
 * @email : ma_zhengyao@163.com
 * @date 2020/3/20
 * @file : main.js
 * @software: PyCharm
*/


/* global $ */
// 全局变量
// index_array: 图表的x轴
var index_array = [0,1,2,3,4,5,6,7,8,9];
// 线性回归模型的概率分布
var probability_line = [];
// 卷积神经网络模型的概率分布
var probability_cnn = [];
// leNet-5神经网络模型的概率分布
var probability_leNet5 = [];
// 识别结果
var reg_num = 0;
// const $ = require('jquery.min');


// 主函数逻辑
class Main {

    // 构造器
    constructor() {
        // canvas手写画板
        this.canvas = document.getElementById('main');
        // 展示输出画板
        this.input = document.getElementById('input');
        // 设置画板宽度449 = 16 * 28 + 1
        this.canvas.width  = 449;
        // 设置画板长度449 = 16 * 28 + 1
        this.canvas.height = 449;

        // getContext() 方法返回一个用于在画布上绘图的环境
        // 二维
        this.ctx = this.canvas.getContext('2d');
        // 增加事件监听 鼠标按下
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        // 增加事件监听 松开鼠标
        this.canvas.addEventListener('mouseup',   this.onMouseUp.bind(this));
        // 增加事件监听 移动鼠标
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));

        // 手写画板初始化
        this.initialize();
    }

    // 手写画板初始化
    initialize() {
        // 背景颜色
        this.ctx.fillStyle = '#ffffff';
        // fillRect() 方法绘制“已填色”的矩形
        this.ctx.fillRect(0, 0, 449, 449);
        // 线的长度
        this.ctx.lineWidth = 1;
        // 设置笔触
        this.ctx.strokeRect(0, 0, 449, 449);
        // 线的长度
        this.ctx.lineWidth = 0.05;

        //循环 绘制网格
        for (var i = 0; i < 27; i++) {
            // 开始一条路径，或重置当前的路径
            this.ctx.beginPath();
            // 创建路径
            this.ctx.moveTo((i + 1) * 16,   0);
            this.ctx.lineTo((i + 1) * 16, 449);
            // 关闭路径
            this.ctx.closePath();
            // 画布上绘制确切的路径
            this.ctx.stroke();

            // 开始一条路径，或重置当前的路径
            this.ctx.beginPath();
            // 创建路径
            this.ctx.moveTo(  0, (i + 1) * 16);
            this.ctx.lineTo(449, (i + 1) * 16);
            // 关闭路径
            this.ctx.closePath();
            // 画布上绘制确切的路径
            this.ctx.stroke();
        }
        // this.drawInput();
        // 输出到td上
        $('#output td').text('').removeClass('success');
    }

    // 输入画板初始化
    initializeOther() {
        // 输入画板
        var input_ctx = this.input.getContext('2d');
        // 背景颜色
        input_ctx.fillStyle = '#ffffff';
        // fillRect() 方法绘制“已填色”的矩形
        input_ctx.fillRect(0, 0, 140, 140);
        // input_ctx.fillStyle="#ffffff";
        // input_ctx.beginPath();
        // input_ctx.fillRect(0,0,140,140);
        // input_ctx.closePath();

        //循环 绘制网格
        for (var i = 0; i < 27; i++) {
            // 开始一条路径，或重置当前的路径
            input_ctx.beginPath();
            // 创建路径
            input_ctx.moveTo((i + 1) * 16,   0);
            // 关闭路径
            input_ctx.closePath();

            // 开始一条路径，或重置当前的路径
            input_ctx.beginPath();
            // 创建路径
            input_ctx.moveTo(  0, (i + 1) * 16);
            // 关闭路径
            input_ctx.closePath();
        }
    }


    // 鼠标按下事件
    onMouseDown(e) {
        // 默认游标
        this.canvas.style.cursor = 'default';
        // 允许绘制
        this.drawing = true;
        // 获取位置
        this.prev = this.getPosition(e.clientX, e.clientY);
    }


    // 松开鼠标事件
    onMouseUp() {
        // 关闭绘制
        this.drawing = false;
        // this.drawInput();
    }


    // 移动鼠标事件
    onMouseMove(e) {
        // 绘制时
        if (this.drawing) {
            // 获取位置
            var curr = this.getPosition(e.clientX, e.clientY);
            // 设置笔的宽度
            this.ctx.lineWidth = 16;
            // 设置笔的类型
            this.ctx.lineCap = 'round';
            // 开始一条路径，或重置当前的路径
            this.ctx.beginPath();
            // 创建路径
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            // 画布上绘制确切的路径
            this.ctx.stroke();
            // 关闭绘制
            this.ctx.closePath();
            // 当前为最后游标位置
            this.prev = curr;
        }
    }


    // 获取定位函数
    getPosition(clientX, clientY) {
        // getBoundingClientRect用于获取某个元素相对于视窗的位置集合
        var rect = this.canvas.getBoundingClientRect();

        // 返回当前的x和y的位置
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }


    // 绘制输入画板和传递数据
    drawInput() {

        // 线性回归模型的概率分布
        probability_line = [];
        // 卷积神经网络模型的概率分布
        probability_cnn = [];
        // leNet-5神经网络模型的概率分布
        probability_leNet5 = [];

        // 获取输入画板的绘制环境
        // 二维
        var ctx = this.input.getContext('2d');
        // 创建新的图像
        var img = new Image();
        // 获取识别结果数字元素
        var reg = document.getElementById('p_class_reg');

        // img图像加载函数
        img.onload = () => {
            // 定义输出数据
            var inputs = [];
            // 创建canvas对象,并获取绘制环境
            var small = document.createElement('canvas').getContext('2d');

            // 缩小画板
            // drawImage() 方法在画布上绘制图像、画布或视频
            // 缩小图片为784 = 28 * 28
            // 后台模型的输入大小为784
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
            // 获取图片的数据
            var data = small.getImageData(0, 0, 28, 28).data;

            // 循环遍历每个像素点
            // 此操作为组织输出数据集
            for (var i = 0; i < 28; i++) {
                for (var j = 0; j < 28; j++) {
                    // red=data[0];
                    // green=data[1];
                    // blue=data[2];
                    // alpha=data[3];
                    // 因为转换为的img数据,有四个通道 所以*4
                    var n = 4 * (i * 28 + j);
                    // 组织输出数据集  一维 , 取出前三个通道的值,取平均,变为一个通道
                    // 数值为255之间的数
                    inputs[i * 28 + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;

                    // 输入画板
                    // 绘制的轨迹 canvas对象
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    // fillRect() 方法绘制“已填色”的矩形 此处为黑色
                    // 将一个像素放大成5*5矩形
                    // 近似28*28 -> 140*140
                    ctx.fillRect(j * 5, i * 5, 5, 5);
                }
            }

            // ...浅拷贝
            // 图像为空白,直接返回
            if (Math.min(...inputs) === 255) {
                return;
            }


            // 接口调用,ajax处理
            $.ajax({
                url: '/api/mnist',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(inputs),
                success: (data) => {
                    // 成功 接受到了flask接口传回的数据
                    // 开始处理展示数据
                    // 将一个 JSON 字符串转换为对象
                    data = JSON.parse(data);

                    // 处理三种模型的数据
                    for (let i = 0; i < 3; i++) {
                        // 最大的概率数
                        var max = 0;
                        // 最大数的下标索引  0-9的哪个数字
                        var max_index = 0;
                        // 处理传过来的数据 每个数组长度为10
                        for (let j = 0; j < 10; j++) {
                            // 取整数 +0.5再 向下取整
                            var value = Math.round(data.results[i][j] * 1000);
                            // 找出最大的值和下标
                            if (value > max) {
                                max = value;
                                max_index = j;
                            }

                            // 精度
                            var digits = String(value).length;
                            // 前端补0
                            for (var k = 0; k < 3 - digits; k++) {
                                value = '0' + value;
                            }
                            // 拼接数字  三位小数精度
                            var text = '0.' + value;

                            // 归一化为1.000
                            if (value > 999) {
                                text = '1.000';
                            }

                            // 组织线性回归模型的数据
                            if (i === 0){
                                // 控制数组大小为10
                                if (probability_line.length > 10){
                                    probability_line = [text];
                                }
                                else {
                                    // 直接添加到末尾
                                    probability_line.push(text);
                                }
                            }

                            // 组织卷积神经网络模型的数据
                            if (i === 1){
                                // 控制数组大小为10
                                if (probability_cnn.length > 10){
                                    probability_cnn = [text];
                                }
                                else {
                                    // 直接添加到末尾
                                    probability_cnn.push(text);
                                }
                            }

                            // 组织lenet5神经网络模型的数据
                            if (i === 2){
                                // 控制数组大小为10
                                if (probability_leNet5.length > 10){
                                    probability_leNet5 = [text];
                                }
                                else {
                                    // 直接添加到末尾
                                    probability_leNet5.push(text);
                                }
                            }

                            // 将数据输出到tr
                            $('#output tr').eq(j + 1).find('td').eq(i).text(text);
                        }


                        // 线性回归模型的概率分布
                        // 刷新全局变量probability_line的值
                        refreshData(0, probability_line);
                        // 卷积神经网络模型的概率分布
                        // 刷新全局变量probability_cnn的值
                        refreshData(1, probability_cnn);
                        // leNet-5神经网络模型的概率分布
                        // 刷新全局变量probability_leNet5的值
                        refreshData(2, probability_leNet5);

                        // 遍历找出概率最大的值的下标
                        for (let k = 0; k < probability_cnn.length; k++) {
                            if (probability_cnn[reg_num] <= probability_cnn[k]) {
                                reg_num = k;
                            }
                        }

                        // 概率最大的值即为识别出的数字
                        reg.innerText = reg_num;
                        // 清0 防止数据混乱
                        reg_num = 0;

                        // 循环向输出界面中写入结果
                        for (let j = 0; j < 10; j++) {
                            if (j === max_index) {
                                $('#output tr').eq(j + 1).find('td').eq(i).addClass('success');
                            } else {
                                $('#output tr').eq(j + 1).find('td').eq(i).removeClass('success');
                            }
                        }
                    }
                }
            });
        };

        // img来自于手写画板
        img.src = this.canvas.toDataURL();
    }
}

// 刷新数据函数 echarts
function refreshData(index, data){
    // 获取元素
    var option = myChart.getOption();
    // 根据下标设置新数据
    option.series[index].data = data;
    // 设置新数据
    myChart.setOption(option);
}


// 重写按钮逻辑
$(() => {
    var main = new Main();
    // 获取识别结果元素
    var reg = document.getElementById('p_class_reg');
    // 点击事件
    $('#clear').click(() => {
        // 手写画板初始化
        main.initialize();
        // 输入画板初始化
        main.initializeOther();
        // 识别结果置空
        reg.innerText = '';
        // 可视化图标数据置空
        refreshData(0, []);
        refreshData(1, []);
        refreshData(2, []);
    });
});


// 识别按钮逻辑
$(() => {
    var main = new Main();
    // 点击事件
    $('#recognition').click(() => {
        // 触发识别主函数
        main.drawInput();
    });
});


//15篇参考文献，其中两篇英文