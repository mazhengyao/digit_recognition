/**
 * @Description: svg动画
 * @author Damon Ma
 * @email : ma_zhengyao@163.com
 * @date 2020/4/18
 * @file : computer.js
 * @software: PyCharm
*/


// 使用.设置JS中元素的属性和值，而不是CSS中的。
// 也可以在CSS中设置这些


TweenMax.set("#mac", {
  transformOrigin: 'bottom 0%',
  scale: 0 });


TweenMax.set("#ipad", {
  transformOrigin: 'bottom 0%',
  autoAlpha: 0,
  scale: 0 });


TweenMax.set("#phone", {
  autoAlpha: 0,
  transformOrigin: 'bottom 0%',
  scale: 0 });


TweenMax.set("#stuff-on-mac", {
  autoAlpha: 0,
  transformOrigin: 'bottom 0%',
  scale: 0 });


TweenMax.set("#stuff-on-iphone", {
  autoAlpha: 0,
  transformOrigin: 'bottom 0%',
  scale: 0 });


TweenMax.set("#stuff-on-ipad", {
  autoAlpha: 0,
  transformOrigin: 'right 0%',
  scale: 0 });


// 函数允许生成动画并将它们粘在一起
// 记住返回值的类型


// 由于item参数，此函数是可重用的
// 使svg中的几个元素变为可见

const visible = item => {
  let tl = new TimelineMax();
  tl.to(item, .5, {
    scale: 1,
    //autoAlpha是一个GSAPs特殊属性
    // 它将不透明度和可见性合并到一个属性中
    autoAlpha: 1,
    ease: Elastic.easeOut.config(1, 0.75) });

  return tl;
};


const bars = item => {
  let tl = new TimelineMax();
  tl.staggerTo(item, 4, {
    scaleY: 0,
    transformOrigin: 'bottom 0%',
    yoyo: true,
    repeat: -1,
    ease: Power0.easeNone,
    stagger: {
      amount: 1.5 } });


  return tl;
};


const lines = item => {
  let tl = new TimelineMax();
  tl.staggerTo(item, 2, {
    autoAlpha: 0,
    transformOrigin: 'center center',
    yoyo: true,
    repeat: -1,
    ease: Power0.easeNone,
    stagger: {
      amount: 1.5 } });


  return tl;
};


const device = item => {
  let tl = new TimelineMax();
  tl.to(item, 2, {
    transformStyle: "preserve-3d",
    force3D: true,
    y: -10,
    z: -10,
    yoyo: true,
    repeat: -1,
    ease: Power0.easeNone });

  return tl;
};


// 创建主时间线来运行所有动画


// 主时间线
const master = new TimelineMax({ delay: .5 });
master.timeScale(1.5);
master.add('s');
master.
add(visible('#mac'), 's+=1.1').
add(visible('#phone'), 's+1.2').
add(visible('#ipad'), 's+1.3').
add(visible('#stuff-on-mac'), 's+1.4').
add(visible('#stuff-on-iphone'), 's+1.5').
add(visible('#stuff-on-ipad'), 's+1.6').
add(bars('.bar'), 's+1.6').
add(bars('.shade'), 's+1.6').
add(lines('.line'), 's+1.6').
add(lines('.line2'), 's+1.6').
add(device('.device'), 's+1.6').
add(device('.device2'), 's+1.6');

// GSDevTools.create({});