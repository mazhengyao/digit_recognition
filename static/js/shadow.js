/**
 * @Description: 文字阴影
 * @author Damon Ma
 * @email : ma_zhengyao@163.com
 * @date 2020/3/15
 * @file : shadow.js
 * @software: PyCharm
*/


// 获取元素
var text = document.getElementById('p_menu');
var shadow = '';


// 循环给阴影赋值
for(var i = 0; i< 60 ; i++) {
  shadow += (shadow ? ',': '') +
      i*1 +'px ' +
      i*1 +'px 0 lightslategray'
}


// 添加阴影
text.style.textShadow = shadow;