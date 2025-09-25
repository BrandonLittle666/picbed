import datetime
import sys
from types import NotImplementedType
from typing import Callable, Literal, TypedDict, Unpack

import shiboken6
from loguru import logger
from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QSize,
    QTimer,
)
from PySide6.QtGui import (
    QCloseEvent,
    QCursor,
    QFocusEvent,
    QFont,
    QFontMetrics,
    QIcon,
    QKeyEvent,
    QMouseEvent,
    QPalette,
    QPixmap,
    Qt,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class MessageLabelKwargs(TypedDict, total=False):
    """ 消息标签的参数 """
    timeout: int
    close_on_focusOut: bool
    parent: QWidget|None
    width: int | None
    close_previous: bool
    get_focus: bool
    processEvent: bool
    align: Qt.AlignmentFlag | None
    offset: QPoint|tuple[int, int]|None
    word_wrap: bool
    topmost: bool
    delay_display: int | None


class MessageLabel(QWidget):
    """ 用来显示消息的label """
    _intance_count = 0
    _last_instance = None
    def __new__(cls, *args, **kwargs):
        cls._intance_count += 1
        return super().__new__(cls)
    
    def __repr__(self) -> str:
        return f'MessageLabel[{self.intance_index}]'
    
    def __init__(self, parent: QWidget|None = None, topmost: bool = False):
        super().__init__(parent)

        self.setFont(QFont('Microsoft YaHei', 12))
        
        if parent is not None:
            self.setFont(parent.font())
            self.setPalette(parent.palette())
            
        # 记录前一个实例
        self.prevMessage = MessageLabel._last_instance
        MessageLabel._last_instance = self
        if self.prevMessage is not None:
            self.prevMessage.nextMessage = self
        self.nextMessage: MessageLabel|None = None
        
        # 设置窗口类型
        if sys.platform == 'win32':
            self.setWindowFlags(Qt.WindowType.CustomizeWindowHint|Qt.WindowType.Tool) 
        else:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint|Qt.WindowType.Tool)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    
        if topmost:
            self.setWindowFlags(self.windowFlags()|Qt.WindowType.WindowStaysOnTopHint)
    
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setObjectName('MessageLabel')

        # ---------- 初始化窗口 ---------- #
        self.init_ui()
        self.init_var()
        # msglable不接受鼠标事件, 否则在msg含有html时, 鼠标事件会被QLabel拦截而无法传递到父窗口实现拖动
        self.msg_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    
    def init_ui(self):
        if not self.objectName():
            self.setObjectName(u"MessageLabel")
        self.resize(658, 300)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setObjectName(u"main_layout")
        self.setLayout(self.main_layout)

        self.content_widget = QWidget(self)
        self.content_widget.setObjectName(u"content_widget")
        if sys.platform != 'win32':
            palette = self.content_widget.palette()
            widow_color = palette.color(QPalette.ColorRole.Window)
            # 增强阴影扩散效果
            self.content_widget.setStyleSheet(f'''
                .QWidget#content_widget {{
                    background-color: {widow_color.name()};
                    border: 1px solid #444;
                    border-radius: 10px;
                }}
                .QWidget#content_widget:hover {{
                    border: 1px solid #666;
                }}
            ''')
        self.main_layout.addWidget(self.content_widget)

        self.vlayout = QVBoxLayout()
        self.vlayout.setSpacing(1)
        self.vlayout.setContentsMargins(1, 1, 1, 7)
        self.content_widget.setLayout(self.vlayout)
        
        self.hlayout_body = QHBoxLayout()
        self.hlayout_body.setSpacing(0)
        self.hlayout_body.setContentsMargins(10, 0, 0, 0)
        
        self.msg_icon_vlayout = QVBoxLayout()
        self.msg_icon_vlayout.setSpacing(0)
        self.msg_icon_vlayout.setContentsMargins(0, 0, 0, 0)
        self.msg_icon_vlayout.setObjectName(u"verticalLayout")

        self.msg_icon_label = QLabel(self)
        self.msg_icon_label.setObjectName(u"msg_icon_label")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHeightForWidth(self.msg_icon_label.sizePolicy().hasHeightForWidth())
        self.msg_icon_label.setSizePolicy(sizePolicy)
        self.msg_icon_label.setMinimumSize(QSize(45, 45))

        self.msg_icon_vlayout.addWidget(self.msg_icon_label, 0, Qt.AlignmentFlag.AlignTop)

        self.hlayout_body.addLayout(self.msg_icon_vlayout)

        self.msg_label = QLabel(self)
        self.msg_label.setObjectName(u"msg_label")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.msg_label.sizePolicy().hasHeightForWidth())
        self.msg_label.setSizePolicy(sizePolicy1)
        self.msg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.msg_label.setWordWrap(True)

        self.hlayout_body.addWidget(self.msg_label)

        self.hlayout_body.setStretch(1, 1)

        self.vlayout.addLayout(self.hlayout_body)

        self.vlayout.setStretch(1, 1)
        
    def init_var(self):
        self.is_deleted = False      # 标记是否已经被删除
        self._close_on_focusOut = False
        self._close_timeout = 5000  # ms
        self._msg_icon = ':/image/null.svg'
        self.fadeout_duration = 50

        self._max_width = 1440
        self._max_height = 900
        
        self.close_timer = QTimer(self)
        self.close_timer.setSingleShot(True)
        self.close_timer.timeout.connect(self.fadeout)
        self.intance_index = self._intance_count
        self._minimum_hide_time = 300  # ms, 最小隐藏时间, 防止闪烁或者意外失去焦点
        
        self.is_fading_out = False
        self._has_focus_time = datetime.datetime.fromtimestamp(0)
        
        # 移动相关
        self._is_dragging = False
        self._drag_start_win_pos = QPoint()      # 鼠标按下时的窗口位置(相对于屏幕)
        self._drag_start_mouse_pos = QPoint()        # 鼠标按下时的鼠标位置(相对于屏幕)

        self.last_focus_widget: QWidget | None = None
         
    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton and e.modifiers() == Qt.KeyboardModifier.NoModifier and self.canDrag(e.position().toPoint()):
            self.windowHandle().startSystemMove()
            self.close_timer.stop()
            return e.accept()
        return super().mousePressEvent(e)

    def canDrag(self, pos):
        """ whether the position is draggable """
        return self._isDragRegion(pos)

    def _isDragRegion(self, pos: QPoint):
        """ Check whether the position belongs to the area where dragging is allowed """
        return True
    
    def focusInEvent(self, event: QFocusEvent) -> None:
        self._has_focus_time = datetime.datetime.now()
        # logger.debug(f'{self} focusInEvent')
        return super().focusInEvent(event)
    
    def focusOutEvent(self, event: QFocusEvent) -> None:
        if self.on_focus_out():
            return super().focusOutEvent(event)
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.fadeout()
            event.accept()
            return None
        return super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.nextMessage is not None:
            self.nextMessage.prevMessage = None
        else:
            # 说明当前是最后一个实例, 将类的最后一个实例置空
            MessageLabel._last_instance = None
        self.nextMessage = None
        self.prevMessage = None
        super().closeEvent(event)
        self.setParent(None)
        try:
            if self.last_focus_widget is not None:
                self.last_focus_widget.setFocus()
        except:
            pass
        self.deleteLater()
        self.is_deleted = True
    
    def setup_message(self, message: str, 
                     title: NotImplementedType = NotImplementedType,   # 标题, 已经弃用, 因为窗口是没有标题栏的
                     msgIcon: str | None  = None, 
                     timeout: int | None = None, 
                     close_on_focusOut: bool | None = None,
                     parent: QWidget | None = None,
                     max_width: int | None = None,
                     min_width: int | None = None,
                     align: Qt.AlignmentFlag|None = None,
                     offset: QPoint|tuple[int, int]|None = (0, 0),
                     word_wrap: bool = True,
                     ):
        # 设置默认值, 如果传入了值, 则使用传入的值并更新默认值
        if close_on_focusOut is None:
            close_on_focusOut = self._close_on_focusOut
        else:
            self._close_on_focusOut = close_on_focusOut
        if timeout is None:
            timeout = self._close_timeout
        else:
            self._close_timeout = timeout
        if max_width is None:
            max_width = self._max_width
        if min_width is None:
            min_width = 200
        if offset is None:
            offset = QPoint(0, 0)
        elif isinstance(offset, tuple):
            offset = QPoint(*offset)
        if align is None:
            align = Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignHCenter
        
        # # 设置文本换行方式（没必要，因为窗口宽度会自动适应，如果限制了，反而会导致窗口宽度超出屏幕）
        # self.msg_label.setWordWrap(word_wrap)

        # 调整窗口大小
        fontmetrics = QFontMetrics(self.msg_label.font())
        width_without_msg_label = (
            (self.msg_icon_label.minimumWidth() if msgIcon is not None else 0) # *1.5是根据下文来的
            +self.hlayout_body.contentsMargins().left()
            +self.hlayout_body.contentsMargins().right()
            +self.hlayout_body.spacing()
            +self.vlayout.contentsMargins().left()
            +self.vlayout.contentsMargins().right())
        flag = Qt.TextFlag.TextWordWrap if word_wrap else Qt.TextFlag.TextSingleLine
        bbox = fontmetrics.boundingRect(0, 0, (max_width - width_without_msg_label), 0, flag, message)
        extra_width = 16
        
        # 窗口宽度
        target_width = bbox.width()+width_without_msg_label+extra_width
        if target_width > self._max_width:
            # 插入零宽空格，以便可以在任意位置换行
            message = '\u200B'.join(message)
            self.msg_label.setText(message)
            # 重新计算boundingRect
            bbox = fontmetrics.boundingRect(0, 0, (max_width - width_without_msg_label), 0, flag, message)
        else:
            self.msg_label.setText(message)
        target_width = max(min_width, min(target_width, (parent.width() if parent is not None else 0xffff), self._max_width))
        
        # 窗口高度
        target_height = max(30, bbox.height(), ((self.msg_icon_label.height() if self.msg_icon_label.isVisible() else 0)))
        target_height += self.vlayout.contentsMargins().top() + self.vlayout.contentsMargins().bottom()
        target_height = min(target_height, self._max_height)
        
        self.resize(target_width, target_height)
        
        # 设置对齐方式, 只有一行时居中, 多行时左对齐
        if bbox.height() > (fontmetrics.height() * 1.5):
            # 多行
            self.msg_label.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
            msgiconsize = fontmetrics.height() * 2.0
        else:
            # 单行
            if msgIcon is not None:
                # 有图标的话靠近图标(左对齐)
                self.msg_label.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
            else:
                self.msg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msgiconsize = fontmetrics.height() * 1.5
        if msgIcon is not None:
            # self.setWindowIcon(QIcon(msgIcon))
            self.msg_icon_label.setPixmap(QPixmap(msgIcon).scaled(QSize(int(msgiconsize), int(msgiconsize)), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.msg_icon_label.show()
        else:
            self.setWindowIcon(QIcon())
            self.msg_icon_label.clear()
            self.msg_icon_label.hide()
        # 设置位置
        self._close_on_focusOut = close_on_focusOut
        if timeout > 0:
            self.close_timer.start(timeout)
        if not self.isVisible():
            # 如果窗口不可见, 说明是新建的对话框
            if parent is not None and parent.isVisible():
                # 默认以父窗口为基准
                rect = parent.geometry()
                startPos = parent.mapToGlobal(QPoint(0, 0))
            else:
                # 没有父窗口, 则使用屏幕的矩形
                screen = QApplication.screenAt(QCursor.pos())
                if not screen:
                    screen = QApplication.primaryScreen()
                rect = screen.availableGeometry()
                startPos = rect.topLeft()
            # 调整位置(默认置顶&水平居中)
            if align & Qt.AlignmentFlag.AlignTop:  # 置顶
                top = startPos.y() + offset.y()
            if align & Qt.AlignmentFlag.AlignBottom:  # 置底
                top = startPos.y() + rect.height() - self.height() + offset.y()
            if align & Qt.AlignmentFlag.AlignVCenter:  # 垂直居中
                top = startPos.y() + (rect.height() - self.height())//2 + offset.y()
            if align & Qt.AlignmentFlag.AlignLeft:   # 左侧
                left = startPos.x() + offset.x()
            if align & Qt.AlignmentFlag.AlignRight:   # 右侧
                left = startPos.x() + rect.width() - self.width() + offset.x()
            if align & Qt.AlignmentFlag.AlignHCenter:  # 水平居中
                left = startPos.x() + (rect.width() - self.width())//2 + offset.x()
            # 防止窗口超出屏幕
            left = left if left > 0 else 0
            top = top if top > 0 else 0
            # 移动到指定位置
            self.move(QPoint(left, top))
        return self

    def moveToAnimation(self, start: QPoint|None = None, end: QPoint|None = None, duration: int = 1000):
        """ 出现效果 """
        # 创建一个动画
        self.move_animation = QPropertyAnimation(self, b"pos")
        self.move_animation.setDuration(duration)  # 动画持续时间：1000毫秒（1秒）
        self.move_animation.setStartValue(start)  # 动画开始位置：(0, 0)
        self.move_animation.setEndValue(end)  # 动画结束位置：当前窗口位置
        # 开始动画
        self.move_animation.start()  
        return self

    def opacityAnimation(self, start: float = 0, end: float = 1, duration: int = 1000, callback: Callable|None = None):
        """ 透明度动画 """
        # 创建一个淡入动画
        self.setWindowOpacity(start)
        if sys.platform == 'win32':
            self.fade_animate = QPropertyAnimation(self, b"windowOpacity")
            self.fade_animate.setDuration(duration)
            self.fade_animate.setStartValue(start)
            self.fade_animate.setEndValue(end)
            self.fade_animate.setEasingCurve(QEasingCurve.InQuad)
            if callback is not None:
                self.fade_animate.finished.connect(callback)
            self.fade_animate.start()
        else:
            # 非windows系统, 禁用
            QTimer.singleShot(duration, callback)
        return self

    def on_focus_out(self, force = False):
        ''' 窗口失去焦点时的回调函数 '''
        if not shiboken6.isValid(self):
            return True
        if ((not force) # 强制关闭时, 忽略最小隐藏时间
           and ((datetime.datetime.now() - self._has_focus_time).total_seconds() * 1000 < self._minimum_hide_time)):
            # 如果窗口失去焦点时间小于最小隐藏时间, 则不关闭
            return True
        if datetime.datetime.now() - self._has_focus_time < datetime.timedelta(seconds=0.2):
            # 忽略100ms内的失去焦点事件, 防止闪烁
            return True
        if self._close_on_focusOut:
            # 关闭条件1: 点击窗口外
            if isinstance(p:=(self.parent()), QWidget) and (not p.isActiveWindow()) and p.isVisible():
                # 如果父窗口不是活动窗口, 则不关闭(点击了应用以外的区域导致的焦点丢失); 但如果父窗口不可见, 则关闭
                return False
            self.fadeout()
        elif self._close_timeout > 0:
            # 关闭条件2: 超时
            self.close_timer.start(self._close_timeout)
        return True

    def fadeout(self, duration: int|None = None):
        """ 淡出 """
        if self.is_deleted or self.is_fading_out:
            return self
        elif not shiboken6.isValid(self):
            return
        elif not self.isVisible():
            self.close()
            return self
        if duration is None:
            duration = self.fadeout_duration
        self.close_timer.stop()
        self._close_on_focusOut = False # 防止再次由focusOutEvent出发fadeout
        self.opacityAnimation(1, 0, duration, self.close)
        self.is_fading_out = True
        return self
    
    def moveByAnimation(self, distance: QPoint, duration: int = 1000):
        """ 移动动画 """
        start = self.pos()
        end = start + distance
        return self.moveToAnimation(start, end, duration)

    def show_forground(self, get_focus=True, processEvent: bool = False, delay_display: int | None = None):
        # 必须激活窗口, 否则_close_on_focusOut无法生效, 因为窗口不会获取焦点
        if isinstance(delay_display, int) and delay_display > 0:
            QTimer.singleShot(delay_display, lambda get_focus=get_focus, processEvent=processEvent: self.show_forground(get_focus, processEvent))
            return self
        if not shiboken6.isValid(self):
            return
        if not self.isVisible():
            self.showNormal()   # 如果窗口不可见, 先显示窗口, 再置顶, 否则置顶无效
        self.show()
        if get_focus or (isinstance(p:=(self.parent()), QWidget) and (not p.isActiveWindow())):
            self.activateWindow()
            self.raise_()
            self.setFocus()
        self._has_focus_time = datetime.datetime.now()
        if processEvent:
            QApplication.processEvents()
        return self
    
    def close_previous(self):
        try:
            if self.prevMessage is None:
                return
            if not shiboken6.isValid(self.prevMessage):
                self.prevMessage = None
            elif self.prevMessage.isWidgetType():
                self.prevMessage.close()
        except Exception as e:
            pass
            logger.debug(f'close_previous error: {e}')
        return self
    
    @staticmethod
    def normal(message: str, 
               timeout: int = 3000, 
               close_on_focusOut=True, 
               parent: QWidget|None = None, 
               width: int|None = None, 
               close_previous=True, 
               get_focus=True, 
               processEvent: bool = False,
               align: Qt.AlignmentFlag|None = None, 
               offset: QPoint|tuple[int, int]|None = None,
               word_wrap: bool = True,
               topmost: bool = False):
        label = MessageLabel(parent=parent, topmost=topmost)
        label.setup_message(message, 
                            # title='提示',
                            timeout=timeout, 
                            close_on_focusOut=close_on_focusOut, 
                            max_width=width,
                            min_width=width,
                            msgIcon=None, 
                            parent=parent,
                            align=align,
                            offset=offset, 
                            word_wrap=word_wrap)
        if close_previous:
            label.close_previous()
        return label.show_forground(get_focus, processEvent=processEvent)
    
    @staticmethod
    def info(message: str, 
             timeout: int = 3000, 
             close_on_focusOut=True, 
             parent: QWidget|None = None, 
             width: int|None = None, 
             close_previous=True, 
             get_focus=True, 
             processEvent: bool = False,
             align: Qt.AlignmentFlag|None = None, 
             offset: QPoint|tuple[int, int]|None = None,
             word_wrap: bool = True,
             delay_display: int | None = None,
             topmost: bool = False):
        label = MessageLabel(parent=parent, topmost=topmost)
        label.setup_message(message, 
                            # title='提示',
                            timeout=timeout, 
                            close_on_focusOut=close_on_focusOut, 
                            max_width=width,
                            min_width=width,
                            msgIcon=':/image/info.svg', 
                            parent=parent,
                            align=align,
                            offset=offset,
                            word_wrap=word_wrap)
        if close_previous:
            label.close_previous()
        return label.show_forground(get_focus, processEvent=processEvent, delay_display=delay_display)
    
    @staticmethod
    def warning(message: str, 
                timeout: int = 0, 
                close_on_focusOut=True, 
                parent: QWidget|None = None, 
                width: int|None = None, 
                close_previous=True, 
                get_focus=True, 
                processEvent: bool = False,
                align: Qt.AlignmentFlag|None = None, 
                offset: QPoint|tuple[int, int]|None = None,
                word_wrap: bool = True,
                topmost: bool = False):
        label = MessageLabel(parent=parent, topmost=topmost)
        label.setup_message(message, 
                            # title='警告',
                            timeout=timeout, 
                            close_on_focusOut=close_on_focusOut, 
                            max_width=width,
                            min_width=width,
                            msgIcon=':/image/warning.svg', 
                            parent=parent,
                            align=align,
                            offset=offset,
                            word_wrap=word_wrap)
        if close_previous:
            label.close_previous()
        return label.show_forground(get_focus, processEvent=processEvent)
    
    @staticmethod
    def error(message: str, 
              timeout: int = 0, 
              close_on_focusOut=True, 
              parent: QWidget|None = None, 
              width: int|None = None, 
              close_previous=True, 
              get_focus=True, 
              processEvent: bool = False,
              align: Qt.AlignmentFlag|None = None, 
              offset: QPoint|tuple[int, int]|None = None,
              word_wrap: bool = True,
              topmost: bool = False,
              delay_display: int | None = None):
        label = MessageLabel(parent=parent, topmost=topmost)
        label.setup_message(message, 
                            # title='错误',
                            timeout=timeout, 
                            close_on_focusOut=close_on_focusOut, 
                            max_width=width,
                            min_width=width,
                            msgIcon=':/image/error.svg', 
                            parent=parent,
                            align=align,
                            offset=offset,
                            word_wrap=word_wrap)
        if close_previous:
            label.close_previous()
        return label.show_forground(get_focus, processEvent=processEvent, delay_display=delay_display)

    @staticmethod
    def show_message(msg_type: Literal['INFO', 'WARNING', 'ERROR', 'NORMAL'], message: str, /, **kwargs: Unpack[MessageLabelKwargs]):
        msg_type = msg_type.upper()
        if msg_type == 'INFO':
            return MessageLabel.info(message, **kwargs)
        elif msg_type == 'WARNING':
            return MessageLabel.warning(message, **kwargs)
        elif msg_type == 'ERROR':
            return MessageLabel.error(message, **kwargs)
        elif msg_type == 'NORMAL':
            return MessageLabel.normal(message, **kwargs)
        else:
            raise ValueError(f'msg_type value {msg_type} is not supported')

