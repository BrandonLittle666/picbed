## 通过共享内存单实例启动
"""
Windows单实例应用程序管理模块

提供基于共享内存的单实例检测和进程间参数传递功能。
支持通过WM_COPYDATA消息传递启动参数给已存在的实例。
"""

import argparse
import pickle
import sys
import tempfile
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

from loguru import logger

APP_KEY = 'picbed'
ARGS_TEMP_PKL_FILE_NAME = 'picbed_args.tmp'


def temp_dir():
    return  Path(tempfile.gettempdir())


def parse_startup_args():
    """
    解析启动参数并返回字典
    
    Returns:
        Dict[str, Any]: 解析后的参数字典
    """
    # 创建主解析器
    parser = argparse.ArgumentParser(description='文件上传下载工具')
    
    # 创建子解析器容器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 创建upload子命令解析器
    upload_parser = subparsers.add_parser('upload', help='上传文件')
    upload_parser.add_argument('files', nargs='+', help='要上传的文件列表')
    upload_parser.add_argument('--overwrite', action='store_true', 
                              help='如果文件已存在则覆盖')
    upload_parser.add_argument('--timeout', type=int, default=30, 
                              help='上传超时时间(秒)，默认30秒')
    
    # 创建download子命令解析器
    download_parser = subparsers.add_parser('download', help='下载文件')
    download_parser.add_argument('urls', nargs='+', help='要下载的URL列表')
    download_parser.add_argument('--outdir', default='./downloads', 
                                help='下载文件保存目录，默认当前目录下的downloads')
    download_parser.add_argument('--resume', action='store_true', 
                                help='支持断点续传')
    download_parser.add_argument('--max-speed', type=int, 
                                help='最大下载速度(KB/s)')


    # 是否忽略单实例检查
    parser.add_argument('--ignore-singleton', action='store_true', 
                        help='忽略单实例检查')
    
    # 日志设置
    parser.add_argument('--logfile-level', type=lambda x: x.upper(), default='DEBUG', choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='文件日志级别')
    parser.add_argument('--logcli-level', type=lambda x: x.upper(), default='DEBUG', choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='控制台日志级别')

    # 解析命令行参数
    args, _ = parser.parse_known_args()

    return args



def initSingleton():
    """ 
    初始化单实例
    
    Returns:
        tuple: (SharedMemory对象, 是否已有实例运行)
    """
    # 解析启动参数
    args = parse_startup_args()

    # 检查是否忽略单实例检查
    if args.ignore_singleton:
        __has_running_instance = False
        logger.info("忽略单实例检查，允许多实例运行")
        return None, __has_running_instance
    
    if not sys.platform == 'win32':
        logger.warning('当前平台不是Windows，无法使用单实例检测')
        return None, False

    import win32api
    import win32gui

    # 注册自定义消息
    NEW_INSTANCE_MESSAGE = win32api.RegisterWindowMessage(APP_KEY)

    # 通过共享内存实现单实例检测
    __has_running_instance = False
    try:
        shm = SharedMemory(name=APP_KEY, create=True, size=8)
        __has_running_instance = False
        logger.debug("创建新的共享内存实例")
    except FileExistsError:
        # 如果共享内存已经存在，说明已经有一个实例
        shm = SharedMemory(name=APP_KEY)
        __has_running_instance = True
    
    if __has_running_instance:
        # 获取已运行实例的窗口句柄
        hwnd = int.from_bytes(shm.buf[:4], byteorder='little')
        logger.debug(f"已存在实例的窗口句柄: {hwnd}")
        if hwnd > 0:
            tempfile = temp_dir() / ARGS_TEMP_PKL_FILE_NAME
            try:
                # 将参数写入临时文件作为备用方案
                tempfile.write_bytes(pickle.dumps(args))
                logger.info(f"参数已写入临时文件: {tempfile}")
                
                # 发送通知消息
                if hwnd:
                    win32gui.PostMessage(hwnd, NEW_INSTANCE_MESSAGE, 0, 0)
                else:
                    # 等待窗口句柄可用
                    start_time = time.time()
                    while time.time() - start_time < 2:  # 增加等待时间到2秒
                        hwnd = int.from_bytes(shm.buf[:4], byteorder='little')
                        if hwnd:
                            win32gui.PostMessage(hwnd, NEW_INSTANCE_MESSAGE, 0, 0)
                            logger.info("通过PostMessage发送新实例通知")
                            break
                        time.sleep(0.1)
                    else:
                        logger.error(f'无法找到已运行的实例窗口句柄，参数: {args}')
            except Exception as e:
                logger.error(f"写入临时文件失败: {e}")
        else:
            __has_running_instance = False
    
    return shm, __has_running_instance


def set_window_handle(shm: SharedMemory, hwnd: int, force: bool = False):
    """
    将窗口句柄写入共享内存
    
    Args:
        shm: 共享内存对象
        hwnd: 窗口句柄
        force: 是否强制写入
    """
    try:
        if not shm or (not hwnd and not force):
            return
        hwnd_bytes = hwnd.to_bytes(4, byteorder='little')
        shm.buf[:4] = hwnd_bytes
        logger.debug(f"窗口句柄已写入共享内存: {hwnd}")
    except Exception as e:
        logger.error(f"写入窗口句柄失败: {e}")


def cleanup_singleton(shm: SharedMemory):
    """
    清理单实例资源
    
    Args:
        shm: 共享内存对象
    """
    try:
        shm.close()
        shm.unlink()
        logger.debug("共享内存已清理")
    except Exception as e:
        logger.debug(f"清理共享内存时出错: {e}")


def get_shm() -> SharedMemory | None:
    """
    获取共享内存对象
    """
    try:
        return SharedMemory(name=APP_KEY)
    except FileNotFoundError:
        return SharedMemory(name=APP_KEY, create=True, size=8)
    except Exception as e:
        logger.error(f"获取共享内存对象失败: {e}")
        return None

