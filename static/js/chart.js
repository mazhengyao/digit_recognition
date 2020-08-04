/**
 * @Description: echarts参数
 * @author Damon Ma
 * @email : ma_zhengyao@163.com
 * @date 2020/4/16
 * @file : chart.js
 * @software: PyCharm
*/


// 基于准备好的dom
// 初始化echarts实例
var myChart = echarts.init(document.getElementById('chart'),'shine');


// 指定图表的配置项和数据
var option = {
    title: {
        text: ' '
    },
    tooltip: {},
    legend: {
        data:['逻辑回归','改进LeNet','LeNet-5']
    },
    xAxis: {
        data: index_array
    },
    yAxis: {},
    series: [
        {
        name: '逻辑回归',
        type: 'bar',
        data: []
        },
        {
            name: '改进LeNet',
            type: 'bar',
            data: []
        },
        {
            name: 'LeNet-5',
            type: 'bar',
            data: []
        }
        ]
};


// 使用刚指定的配置项和数据显示图表。
myChart.setOption(option);
