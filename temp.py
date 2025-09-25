import sys
from PySide6.QtCore import Qt, QUrl, QSize
from PySide6.QtGui import QPainter, QTransform, QWheelEvent
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频播放器")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建媒体播放器
        self.player = QMediaPlayer()
        self.player.playbackStateChanged.connect(self.update_buttons)
        
        # 创建视频窗口
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(640, 480)
        self.video_widget.setStyleSheet("background-color: black;")
        
        # 将视频窗口与播放器关联
        self.player.setVideoOutput(self.video_widget)
        
        # 创建滚动区域以支持缩放和拖拽
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.video_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setStyleSheet("QScrollArea { background-color: black; }")
        
        # 创建控制按钮
        self.open_btn = QPushButton("打开视频")
        self.play_btn = QPushButton("播放")
        self.pause_btn = QPushButton("暂停")
        self.stop_btn = QPushButton("停止")
        
        # 初始禁用播放控制按钮
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        # 连接信号和槽
        self.open_btn.clicked.connect(self.open_file)
        self.play_btn.clicked.connect(self.player.play)
        self.pause_btn.clicked.connect(self.player.pause)
        self.stop_btn.clicked.connect(self.player.stop)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)
        
        # 布局管理
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.open_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.stop_btn)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll_area)
        main_layout.addLayout(control_layout)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def open_file(self):
        """打开视频文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            print(f"选择的视频文件: {file_path}")
            # 设置媒体源
            self.player.setSource(QUrl.fromLocalFile(file_path))
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            # 自动播放
            self.player.play()
    
    def on_media_status_changed(self, status):
        """媒体状态变化时的处理"""
        from PySide6.QtMultimedia import QMediaPlayer
        
        status_names = {
            QMediaPlayer.MediaStatus.NoMedia: "NoMedia",
            QMediaPlayer.MediaStatus.LoadingMedia: "LoadingMedia", 
            QMediaPlayer.MediaStatus.LoadedMedia: "LoadedMedia",
            QMediaPlayer.MediaStatus.BufferingMedia: "BufferingMedia",
            QMediaPlayer.MediaStatus.BufferedMedia: "BufferedMedia",
            QMediaPlayer.MediaStatus.EndOfMedia: "EndOfMedia",
            QMediaPlayer.MediaStatus.InvalidMedia: "InvalidMedia"
        }
        
        print(f"媒体状态变化: {status_names.get(status, 'Unknown')}")
        
        # 当媒体加载完成后调整视频窗口大小
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            # 获取视频的实际尺寸
            video_size = self.player.duration()
            if video_size > 0:
                # 设置视频窗口的合适大小
                self.video_widget.resize(640, 480)
                print(f"视频加载完成，时长: {video_size}ms")
    
    def update_buttons(self, state):
        """根据播放状态更新按钮状态"""
        from PySide6.QtMultimedia import QMediaPlayer
        
        state_names = {
            QMediaPlayer.PlaybackState.StoppedState: "Stopped",
            QMediaPlayer.PlaybackState.PlayingState: "Playing", 
            QMediaPlayer.PlaybackState.PausedState: "Paused"
        }
        
        print(f"播放状态变化: {state_names.get(state, 'Unknown')}")
        
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
        elif state == QMediaPlayer.PlaybackState.StoppedState:
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 确保中文显示正常
    font = app.font()
    font.setFamily("SimHei")
    app.setFont(font)
    
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())
    