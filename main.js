/**
 * @Description: electron启动配置文件
 * @author Damon Ma
 * @email : ma_zhengyao@163.com
 * @date 2020/3/28
 * @file : main.js
 * @software: PyCharm
*/

// outside { globalShortcut, Menu } from 'electron';

// 获取electron的窗口菜单等元素
const { Menu, app, BrowserWindow } = require("electron");
// 获取url
const url = require('url');
// 获取路径
const path = require('path');
// 获取electron主程序
const electron = require('electron');

// 获取MenuItem、ipcMain
// const MenuItem = require('MenuItem');
// const ipc = require('ipcMain');


// 用 Tray 来表示一个图标,这个图标处于正在运行的系统的通知区
// 通常被添加到一个 context menu 上
const Tray = electron.Tray;

// 系统托盘对象
var appTray = null;

// 自定义菜单栏
var template = [{
    label: '编辑',
    submenu: [{
        label: '撤销',
        accelerator: 'CmdOrCtrl+Z',
        role: 'undo'
    }, {
        label: '重做',
        accelerator: 'Shift+CmdOrCtrl+Z',
        role: 'redo'
    }, {
        type: 'separator'
    }, {
        label: '复制',
        accelerator: 'CmdOrCtrl+C',
        role: 'copy'
    }, {
        label: '粘贴',
        accelerator: 'CmdOrCtrl+V',
        role: 'paste'
    }]
}, {
    label: '帮助',
    role: 'help',
    submenu: [{
        label: '学习更多',
        click: function () {
            electron.shell.openExternal('http://electron.atom.io')
        }
    }]
}];

// 主程序：创建窗口
function createWindow() {
    // 窗口变量
    // 定义宽度和高度
    // 完全支持node
    let win = new BrowserWindow({
        width: 1245,
        height: 600,
        webPreferences: {
            // preload: path.join(__dirname, 'main.js'),
            nodeIntegration: true
        }
    });

    // 设置自定义的菜单栏
    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);

    //系统托盘右键菜单
    var trayMenuTemplate = [
        {
            //打开相应页面
            label: '设置',
            click: function () {}
        },
        {
            label: '帮助',
            click: function () {}
        },
        {
            label: '关于',
            click: function () {}
        },
        {
            label: '退出',
            click: function () {
                //ipc.send('close-main-window');
                app.quit();
            }
        }
    ];

    // 可以通过文件来定义加载的主界面
    // and load the canvas.html of the app.
    // win.loadURL(url.format({
    //     pathname: path.join(__dirname, 'templates/index.html'),
    //     protocol: 'file:',
    //     slashes: true
    // }));

    //去掉默认菜单栏
    Menu.setApplicationMenu(null);

    // eslint-disable-next-line no-console
    // 打印启动信息
    console.log('mainWindow opened');

    // win.loadFile("templates/welcome.html");
    // 通过url来加载定义的主界面
    win.loadURL('http://127.0.0.1:8000/welcome');

    // 打开开发者工具
    // win.webContents.openDevTools({mode:'bottom'});

    // 取消对窗口对象的引用
    win.on('closed', function () {
        // Dereference the window object, usually you would store windows
        // in an array if your app supports multi windows, this is the time
        // when you should delete the corresponding element.
        win = null;
    });
}

// Emitted when the window is closed.
// 窗口关闭时触发的动作

// 可以定义 运行python文件
// 从 python-shell 导入一个 PythonShell 对象 (注意大小写)
// const {PythonShell}  = require("python-shell");
// PythonShell 主要有 run() 和 runString() 两个方法, 这里用 run()
// run() 第一个参数是要调用的 py 文件的路径
// 第二个参数是可选配置 (一般传 null)
// 第三个参数是回调函数
// PythonShell.run(
// 	"main.py", null, function (err, results) {
//         if (err)
//             throw err;
//         console.log('main.py running');
//         console.log('results', results);
//     }
// );


// 定义 窗口启动
app.on("ready", createWindow);

// 定义 窗口关闭
app.on('window-all-closed', () => {
    // 在 macOS 上，除非用户用 Cmd + Q 确定地退出，
    // 否则绝大部分应用及其菜单栏会保持激活。
    if (process.platform !== 'darwin') {
        app.quit()
    }
});


// app.on('activate', () => {
//     // 在macOS上，当单击dock图标并且没有其他窗口打开时，
//     // 通常在应用程序中重新创建一个窗口。
//     if (win === null) {
//       createWindow()
//     }
// });


// 在这个文件中，你可以续写应用剩下主进程代码。
// 也可以拆分成几个文件，然后用 require 导入。