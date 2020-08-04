/**
 * @Description: nav导航栏
 * @author Damon Ma
 * @email : ma_zhengyao@163.com
 * @date 2020/4/15
 * @file : nav.js
 * @software: PyCharm
*/


// 获取元素
const indicator = document.querySelector('.nav-indicator');
const items = document.querySelectorAll('.nav-item');


function handleIndicator(el) {
  // 遍历
  items.forEach(item => {
    // 移除状态
    item.classList.remove('is-active');
    item.removeAttribute('style');
  });

  indicator.style.width = `${el.offsetWidth}px`;
  indicator.style.left = `${el.offsetLeft}px`;
  indicator.style.backgroundColor = el.getAttribute('active-color');

  // 添加状态
  el.classList.add('is-active');
  el.style.color = el.getAttribute('active-color');
}


// 遍历
items.forEach((item, index) => {
  item.addEventListener('click', e => {handleIndicator(e.target);});
  item.classList.contains('is-active') && handleIndicator(item);
});