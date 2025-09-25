# PicBed

一个基于 S3 存储的兼容 `PicGo`的图床管理工具，使用 PySide6 构建的桌面应用程序。

## 开发动机

为什么不用 `PicGo` ?

- 尝试过PicGo, 使用minio插件连接自己搭建的图床, 不知为何上传卡顿, 即使是局域网内也要反应一会儿
- 对PicGo的技术栈不熟, 不会从源码修改
- 至少在开发者的机器上, 上传响应比最新版的 `PicGo`迅速

## 功能特性

- 🖼️ **图片上传管理** - 支持多种图片格式（PNG、JPG、GIF、WebP、BMP、TIFF、SVG、ICO）
- ☁️ **S3 兼容存储** - 支持 AWS S3 及兼容的对象存储服务
- 🎨 **界面** - 基于 PySide6
- 📋 **剪贴板集成** - 支持从剪贴板直接上传图片
- 🌐 **HTTP 服务器** - 内置 PicGo 兼容的 HTTP 上传接口
- 🔄 **自动重命名** - 智能文件命名和冲突处理
- 📊 **缓存管理** - 高效的图片缓存和预览系统
- 🚀 **单例模式**

## 系统要求

- Windows 10/11
- Python 3.11+
- 支持 S3 兼容的对象存储服务

## 安装

### 从源码安装

1. 下载仓库：

    直接下载整个仓库文件, 解压并进入到解压后的目录

2. 安装依赖：

```bash
uv sync
```

3. 运行应用：

```bash
uv run main.py
```

## 配置

### S3 配置

在应用设置中配置以下参数：

- **Endpoint URL** - S3 服务端点（如：`https://s3.amazonaws.com`）
- **Region** - 存储区域
- **Bucket** - 存储桶名称
- **Access Key** - 访问密钥
- **Secret Key** - 秘密密钥
- **Path Style** - 是否使用路径样式访问
- **Prefix** - 文件路径前缀

### 重命名

支持以下占位符：

- `{year}` - 年份
- `{month}` - 月份
- `{day}` - 日期
- `{hour}` - 小时
- `{minute}` - 分钟
- `{second}` - 秒
- `{uuid}` - UUID
- `{timestamp}` - 时间戳
- `{random}` - 随机字符串

## 使用方法

### 图形界面

1. 启动应用后，在系统托盘中找到 PicBed 图标
2. 右键点击图标打开菜单
3. 选择"设置"配置 S3 参数
4. 选择"上传图片"或使用快捷键上传图片
5. 上传完成后，图片链接会自动复制到剪贴板

### HTTP 接口

应用内置 HTTP 服务器，默认端口 36677，提供 PicGo 兼容的上传接口：

```bash
# 上传图片
curl -X POST http://localhost:36677/upload \
  -F "file=@image.jpg"
```

## 开发

### 构建

使用提供的 PowerShell 脚本构建可执行文件：

```powershell
.\build.ps1
```

### 项目结构

```
picbed/
├── main.py              # 应用入口
├── picbed/
│   ├── app.py           # 主应用程序
│   ├── s3_client.py     # S3 客户端
│   ├── server.py        # HTTP 服务器
│   ├── cache_manager.py # 缓存管理
│   ├── message_label.py # 消息标签组件
│   ├── singleton.py     # 单例模式管理
│   ├── utils.py         # 工具函数
│   └── image/           # 资源文件
└── nuitkaFolder/        # 构建输出目录
```

## 技术栈

- **GUI 框架**: PySide6
- **云存储**: boto3 (AWS SDK)
- **图像处理**: Pillow
- **日志**: loguru
- **构建工具**: Nuitka
- **包管理**: uv

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0

- 初始版本发布
- 支持 S3 兼容存储
- 现代化 PySide6 界面
- HTTP 服务器集成
- 缓存管理系统
