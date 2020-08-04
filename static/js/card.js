/**
 * @Description: 资料卡片
 * @author Damon Ma
 * @email : ma_zhengyao@163.com
 * @date 2020/4/15
 * @file : card.js
 * @software: PyCharm
*/


// card1鼠标经过效果
$('#card1').hover(function () {

    // 移除样式
    if ($(this).hasClass("active")){

        $('#card1 .bottom').slideUp(function () {
            $('#card1').removeClass("active");
        })

    //添加样式
    }else {
            $('#card1').addClass("active");
            // 划出
            $('#card1 .bottom').stop().slideDown();
    }

});


// card2鼠标经过效果
$('#card2').hover(function () {

    // 移除样式
    if ($(this).hasClass("active")){

        $('#card2 .bottom').slideUp(function () {
            $('#card2').removeClass("active");
        })

    //添加样式
    }else {
            $('#card2').addClass("active");
            // 划出
            $('#card2 .bottom').stop().slideDown();
    }

});



// card3鼠标经过效果
$('#card3').hover(function () {

    // 移除样式
    if ($(this).hasClass("active")){

        $('#card3 .bottom').slideUp(function () {
            $('#card3').removeClass("active");
        })

    //添加样式
    }else {
            $('#card3').addClass("active");
            // 划出
            $('#card3 .bottom').stop().slideDown();
    }

});