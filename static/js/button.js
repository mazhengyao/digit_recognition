/**
 * @Description: button 状态转换
 * @author Damon Ma
 * @email : ma_zhengyao@163.com
 * @date 2020/4/12
 * @file : button.js
 * @software: PyCharm
*/


$($('.dynamic')).click(
    function () {
        // 获取对象
        var polyline_checkmark = document.getElementById('polyline_checkmark');
        polyline_checkmark.getAttribute("points");
        // 清空样式
        polyline_checkmark.setAttribute("points","",0);
        // 状态转换
        $(this).toggleClass('active');
        // 延时效果 伪加载
        setTimeout(
            () => {
                // var polyline_checkmark = document.getElementById('polyline_checkmark');
                polyline_checkmark.getAttribute("points");
                polyline_checkmark.setAttribute("points","2,10 12,18 28,2",0);
                $(this).toggleClass('verity');
            },500
        );

        // 状态转换
        // 延时效果 成功
        setTimeout(
            () => {
                // var polyline_checkmark = document.getElementById('polyline_checkmark');
                polyline_checkmark.setAttribute("points","",0);
                // $(this).toggleClass('active');
                $(this).toggleClass('active');
                $(this).toggleClass('verity');
            },1500
        );


    }
);