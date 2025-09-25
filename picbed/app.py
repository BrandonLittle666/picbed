from __future__ import annotations

import datetime
import io
import mimetypes
import os
import pickle
import sys
import threading
import time
import warnings
from argparse import Namespace
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Empty, PriorityQueue
from typing import Any, Callable, Generic, TypeVar
from urllib.error import HTTPError
from urllib.request import urlopen

import psutil
import win32api
from loguru import logger
from PIL import Image
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from uuid_extensions import uuid7

import picbed.res_rc  # type: ignore[import-untyped]
from picbed.cache_manager import (
    ImageCacheManager,
    PreviewData,
    estimate_qicon_size,
    estimate_qpixmap_size,
)
from picbed.message_label import MessageLabel
from picbed.s3_client import S3Client, S3Config
from picbed.server import PicGoLikeServer
from picbed.singleton import (
    APP_KEY,
    ARGS_TEMP_PKL_FILE_NAME,
    get_shm,
    parse_startup_args,
    set_window_handle,
    temp_dir,
)
from picbed.utils import (
    OUTPUT_PLACEHOLDERS,
    SUPPORT_PLACEHOLDERS,
    acquire_lock,
    appdir,
    auto_rename,
    format_size,
    is_image_key,
    read_clipboard_image_bytes,
    render_placeholders,
)

# ignore decompression bomb warning
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class TaskType(Enum):
    UPLOAD = 'upload'
    DOWNLOAD = 'download'
    PREVIEW = 'preview'
    THUMBNAIL = 'thumbnail'


class TaskPriority(Enum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2


@dataclass
class TaskInfo:
    """Information about a background task"""
    task_id: str
    task_type: TaskType
    obj_key: str = ''   # key of the object
    description: str = ''
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    is_registered: bool = False
    s3: S3Client | None = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def __lt__(self, other: 'TaskInfo') -> bool:
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class ResourceMonitor:
    """Monitor system resources to prevent crashes"""
    
    def __init__(self):
        self._last_check = 0.0
        self._check_interval = 5.0  # Check every 5 seconds
        self._memory_threshold = 85  # Warn at 85% memory usage
        self._cpu_threshold = 90     # Warn at 90% CPU usage
        
    def should_limit_tasks(self) -> tuple[bool, str]:
        """Check if we should limit new tasks due to resource constraints"""
        return False, ''
        now = time.time()
        if now - self._last_check < self._check_interval:
            return False, ""
            
        self._last_check = now
        
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self._memory_threshold:
                return True, f"High memory usage: {memory.percent:.1f}%"
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self._cpu_threshold:
                return True, f"High CPU usage: {cpu_percent:.1f}%"
                
            return False, ""
        except Exception as e:
            logger.debug(f"Resource monitoring failed: {e}")
            return False, ""
    
    def get_resource_info(self) -> str:
        """Get current resource usage info"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return f"Memory: {memory.percent:.1f}% CPU: {cpu_percent:.1f}%"
        except Exception:
            return "Resource info unavailable"


class TaskManager:
    """Manages concurrent tasks with limits and prioritization"""
    
    def __init__(self, s3_config: S3Config|None = None, 
                 max_concurrent_uploads: int = 4, max_concurrent_downloads: int = 4, 
                 max_concurrent_previews: int = 3, max_concurrent_thumbnails: int = 2):
        self._s3_config = s3_config
        self.max_concurrent: dict[TaskType, int] = {
            TaskType.UPLOAD: max_concurrent_uploads,
            TaskType.DOWNLOAD: max_concurrent_downloads, 
            TaskType.PREVIEW: max_concurrent_previews,
            TaskType.THUMBNAIL: max_concurrent_thumbnails
        }
        
        # S3Client pool, for client reuse
        self.s3_pool: list[S3Client] = []       
        
        self.active_tasks: dict[TaskType, dict[str, TaskInfo]] = {
            TaskType.UPLOAD: {},
            TaskType.DOWNLOAD: {},
            TaskType.PREVIEW: {},
            TaskType.THUMBNAIL: {}
        }
        self.pending_queues: dict[TaskType, PriorityQueue[TaskInfo]] = {
            TaskType.UPLOAD: PriorityQueue(),
            TaskType.DOWNLOAD: PriorityQueue(),
            TaskType.PREVIEW: PriorityQueue(),
            TaskType.THUMBNAIL: PriorityQueue()
        }
        self._lock = threading.Lock()
        self._shutdown = False
        self._resource_monitor = ResourceMonitor()

    def setup_s3_config(self, s3_config: S3Config|None):
        self._s3_config = s3_config

    def allocate_s3_client(self, task_info: TaskInfo, *, lock_requierd: bool = False) -> bool:
        """Get a S3Client from the pool"""
        if not self._s3_config:
            return False
        with acquire_lock(self._lock, timeout=1.5, already_acquired=lock_requierd) as acquired:
            if not acquired:
                return False
            idle_client = next((client for client in self.s3_pool if not client._is_in_task), None)
            if idle_client:
                idle_client._is_in_task = True
                task_info.s3 = idle_client
            else:
                new_client = S3Client(self._s3_config)
                new_client._is_in_task = True
                self.s3_pool.append(new_client)
                task_info.s3 = new_client
            return True

    def release_s3_client(self, task_info: TaskInfo):
        if not task_info or not task_info.s3:
            return
        task_info.s3._is_in_task = False
        task_info.s3 = None
    
    def can_start_task(self, task_type: TaskType, lock_requierd: bool = False) -> bool:
        """Check if we can start a new task of the given type"""
        with acquire_lock(self._lock, timeout=1.5, already_acquired=lock_requierd) as acquired:
            if not acquired:
                return False
            if self._shutdown:
                return False
            
            # Check resource constraints
            should_limit, reason = self._resource_monitor.should_limit_tasks()
            if should_limit:
                logger.debug(f"Limiting new tasks due to resource constraints: {reason}")
                return False
                
            return len(self.active_tasks[task_type]) < self.max_concurrent[task_type]
    
    def register_task(self, task_info: TaskInfo, lock_requierd: bool = False) -> bool:
        """Register a new task. Returns True if task can start immediately, False if queued."""
        with acquire_lock(self._lock, timeout=1.5, already_acquired=lock_requierd) as acquired:
            if not acquired:
                return False
            if self._shutdown:
                return False
            task_info.is_registered = True
            if self.can_start_task(task_info.task_type, lock_requierd=True) and self.allocate_s3_client(task_info, lock_requierd=True):
                task_info.started_at = time.time()
                self.active_tasks[task_info.task_type][task_info.task_id] = task_info
                logger.debug(f"Started task immediately: {task_info.task_id} ({task_info.task_type})")
                return True
            else:
                self.pending_queues[task_info.task_type].put(task_info)
                logger.debug(f"Queued task: {task_info.task_id} ({task_info.task_type}, {self.pending_queues[task_info.task_type].qsize()})")
                return False
    
    def complete_task(self, task_id: str, task_type: TaskType, lock_requierd: bool = False) -> TaskInfo | None:
        """Mark a task as complete and try to start the next queued task"""
        with acquire_lock(self._lock, timeout=1.5, already_acquired=lock_requierd) as acquired:
            if not acquired:
                return None
            if task_id in self.active_tasks[task_type]:
                st, ft = self.active_tasks[task_type][task_id].started_at, self.active_tasks[task_type][task_id].finished_at
                task_info = self.active_tasks[task_type].pop(task_id)
                self.release_s3_client(task_info)
                logger.debug(f"Completed task: {task_id} ({task_type=}, {st=}, {ft=})")
                
                # Try to start next queued task
                if not self.pending_queues[task_type].empty():
                    try:
                        next_task = self.pending_queues[task_type].get_nowait()
                        if not self.allocate_s3_client(next_task, lock_requierd=True):
                            # No s3 client available, keep waiting
                            self.pending_queues[task_type].put(next_task)
                            return None
                        next_task.started_at = time.time()
                        self.active_tasks[task_type][next_task.task_id] = next_task
                        logger.debug(f"Started queued task: {next_task.task_id} ({task_type})")
                        return next_task
                    except Empty:
                        pass
                return None
            return None
    
    def cancel_task(self, task_id: str, task_type: TaskType, lock_requierd: bool = False) -> bool:
        """Cancel a task (remove from active or pending)"""
        with acquire_lock(self._lock, timeout=1.5, already_acquired=lock_requierd) as acquired:
            if not acquired:
                return False
            if task_id in self.active_tasks[task_type]:
                task_info = self.active_tasks[task_type].pop(task_id)
                self.release_s3_client(task_info)
                logger.debug(f"Cancelled active task: {task_id} ({task_type})")
                return True
            
            # Try to remove from pending queue (this is inefficient but rare)
            temp_items: list[TaskInfo] = []
            found = False
            try:
                while not self.pending_queues[task_type].empty():
                    item = self.pending_queues[task_type].get_nowait()
                    if item.task_id == task_id:
                        found = True
                        logger.debug(f"Cancelled pending task: {task_id} ({task_type})")
                    else:
                        temp_items.append(item)
            except Empty:
                pass
            
            # Put back remaining items
            for item in temp_items:
                self.pending_queues[task_type].put(item)
            return found
    
    def get_status(self) -> dict[TaskType, dict]:
        """Get current status of all task types"""
        with acquire_lock(self._lock, timeout=1.5) as acquired:
            if not acquired:
                return {
                    t: {
                        'active': len(self.active_tasks[t]),
                        'pending': self.pending_queues[t].qsize(),
                        'max_concurrent': self.max_concurrent[t],
                    }
                    for t in self.active_tasks.keys()
                }
            ret = {
                task_type: {
                    'active': len(self.active_tasks[task_type]),
                    'pending': self.pending_queues[task_type].qsize(),
                    'max_concurrent': self.max_concurrent[task_type]
                }
                for task_type in self.active_tasks.keys()
            }
        return ret
    
    def cancel_pending_tasks(self, task_type: TaskType, lock_requierd: bool = False) -> int:
        """Cancel all pending tasks of a specific type"""
        with acquire_lock(self._lock, timeout=1.5, already_acquired=lock_requierd) as acquired:
            if not acquired:
                return 0
            if task_type not in self.pending_queues:
                return 0
                
            cancelled_count = 0
            queue = self.pending_queues[task_type]
            
            # Extract all items and don't put them back
            try:
                while not queue.empty():
                    queue.get_nowait()
                    cancelled_count += 1
            except Empty:
                pass
                
            return cancelled_count
    
    def shutdown(self):
        """Shutdown task manager"""
        with acquire_lock(self._lock, timeout=2.0) as acquired:
            if not acquired:
                self._shutdown = True
                return
            self._shutdown = True
            # Clear all pending queues
            for queue in self.pending_queues.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except Empty:
                        break


WORKER = TypeVar('WORKER', bound=QThread)
class WorkerTracer(Generic[WORKER]):
    def __init__(self, task_info: TaskInfo, worker: WORKER):
        self.task_info = task_info
        self._worker = worker


class GeneralWorker(QThread):
    result = Signal(object)
    failed = Signal(Exception)

    def __init__(self, fn: Callable[[TaskInfo, ...], Any], task_info: TaskInfo, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._task_info = task_info
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:  # type: ignore[override]
        try:
            result = self._fn(self._task_info, *self._args, **self._kwargs)
            self.result.emit(result)
        except Exception as e:  # noqa: BLE001
            logger.exception("Background task failed")
            self.failed.emit(e)
            return


class PreviewWorker(QThread):
    result = Signal(object)
    failed = Signal(Exception)
    blurry = Signal(str, object)
    progress = Signal(str, str, float)

    def __init__(self, key: str, s3: S3Client, viewport_size: QSize):
        super().__init__()
        self._key = key
        self.s3 = s3
        self.viewport_size = viewport_size

    def run(self):
        data: bytes | None = None
        if not is_image_key(self._key):
            self.result.emit(PreviewData(key=self._key, data=None, pixmap=None, size=None, icon=None, is_animated=False, is_error=True))
            return
        url = self.s3.object_url(self._key)
        try:
            with urlopen(url) as resp:  # nosec - presigned URL
                # Download in chunks so we can report progress
                content_length_header = resp.headers.get('Content-Length') or resp.headers.get('content-length')
                total_bytes = int(content_length_header) if content_length_header and content_length_header.isdigit() else 0
                prefer_chunk_size = 512 * 1024
                chunk_size = prefer_chunk_size if total_bytes > prefer_chunk_size else total_bytes
                downloaded_bytes = 0
                chunks: list[bytes] = []
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    downloaded_bytes += len(chunk)
                    if total_bytes > 0:
                        # Map download progress to first half [0.0, 0.5]
                        self.progress.emit('download', self._key, downloaded_bytes / total_bytes)
                data: bytes = b"".join(chunks)

            self.progress.emit('decode', self._key, 1)

            first_img: QPixmap | None = None
            ba = QByteArray(data)
            buf = QBuffer(ba)
            buf.open(QBuffer.OpenModeFlag.ReadOnly)
            reader = QImageReader(buf)
            
            # return first frame and full image if the image is animated
            if reader.supportsAnimation():
                try:
                    img = reader.read()
                    if not img.isNull():
                        first_img = QPixmap.fromImage(img)
                    self.result.emit(PreviewData(key=self._key, data=data, pixmap=first_img, size=reader.size(), icon=None, is_animated=True))
                except Exception:
                    self.failed.emit(Exception("Failed to read animated image"))
                return
            
            if self._key.lower().endswith('.svg'):
                # render svg to pixmap
                pm = QPixmap()
                pm.loadFromData(data)
                self.result.emit(PreviewData(key=self._key, data=None, pixmap=pm, size=reader.size(), icon=None, is_animated=False))
                return

            # decode image via pillow
            img = Image.open(io.BytesIO(data))
            img = img.convert("RGBA")
            data = img.tobytes("raw", "RGBA")
            qimage = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
            pm = QPixmap.fromImage(qimage)

            if not pm.isNull():
                self.result.emit(PreviewData(key=self._key, data=None, pixmap=pm, size=reader.size(), icon=None, is_animated=False))
            else:
                self.failed.emit(Exception("Failed to decode image"))
            return
        except HTTPError as he:
            return self.result.emit(PreviewData(key=self._key, data=data, pixmap=QPixmap(), size=None, icon=None, is_animated=False, is_error=True))

        except Exception as e:
            self.failed.emit(e)


class ImageGraphicsView(QGraphicsView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._message_item = None
        self._movie: QMovie | None = None
        self._movie_bytes: QByteArray | None = None
        self._movie_buffer: QBuffer | None = None
        self._movie_fit_done: bool = False
        self._scale_factor = 1.0
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor( 24, 24, 24))

    def clear(self) -> None:  # type: ignore[override]
        self._scene.clear()
        self._pixmap_item = None
        self._message_item = None
        # Stop and release any active movie
        if self._movie is not None:
            try:
                self._movie.stop()
            except Exception:
                pass
        # 重置场景范围
        self._scene.setSceneRect(QRectF(0, 0, 0, 0))
        self.resetTransform()
        self._center_scene()
        self._movie = None
        self._movie_buffer = None
        self._movie_bytes = None
        self._movie_fit_done = False
        self._scale_factor = 1.0

    def show_message(self, text: str) -> None:
        self.clear()
        self._message_item = QGraphicsTextItem(text)
        self._scene.addItem(self._message_item)
        # set font
        font = self._message_item.font()
        font.setPointSize(11)
        self._message_item.setFont(font)
        self._message_item.setDefaultTextColor(Qt.GlobalColor.lightGray)
        # reset transform
        self.resetTransform()
        self._scene.setSceneRect(self._message_item.boundingRect())

    def set_pixmap(self, pm: QPixmap, update: bool = False) -> None:
        self.clear()
        self._pixmap_item = self._scene.addPixmap(pm)
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.setSceneRect(pm.rect())
        self.resetTransform()
        self._fit_view(pm, self._pixmap_item)
        self._scale_factor = 1.0

    def set_movie(self, data: bytes) -> None:
        # Display an animated image by streaming frames via QMovie
        self.clear()
        self._movie_bytes = QByteArray(data)
        self._movie_buffer = QBuffer(self._movie_bytes)
        try:
            self._movie_buffer.open(QBuffer.OpenModeFlag.ReadOnly)
        except Exception:
            # Fallback: try still image
            pm = QPixmap()
            if pm.loadFromData(data):
                self.set_pixmap(pm)
            else:
                self.show_message("Unsupported animation")
            return
        self._movie = QMovie(self._movie_buffer, b"", self)
        self._movie.setCacheMode(QMovie.CacheMode.CacheAll)

        def _on_frame(_index: int) -> None:
            if self._movie is None:
                return
            pm = self._movie.currentPixmap()
            if pm.isNull():
                return
            if self._pixmap_item is None:
                self._pixmap_item = self._scene.addPixmap(pm)
                self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
                self._scene.setSceneRect(pm.rect())
                self.resetTransform()
                self._fit_view(pm, self._pixmap_item)
                self._scale_factor = 1.0
                self._movie_fit_done = True
            else:
                self._pixmap_item.setPixmap(pm)
                if not self._movie_fit_done:
                    self._fit_view(pm, self._pixmap_item)
                    self._movie_fit_done = True

        self._movie.frameChanged.connect(_on_frame)
        try:
            self._movie.start()
        except Exception:
            # Fallback to still
            pm = QPixmap()
            if pm.loadFromData(data):
                self.set_pixmap(pm)
            else:
                self.show_message("Failed to play animation")

    def fit_if_needed(self) -> None:
        if self._pixmap_item is None:
            return
        self._fit_view(self._pixmap_item.pixmap(), self._pixmap_item)

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._pixmap_item is None:
            return super().wheelEvent(event)
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # Zoom toward mouse position
        old_pos = self.mapToScene(event.position().toPoint())
        zoom_in = 1.25
        zoom_out = 0.8
        factor = zoom_in if delta > 0 else zoom_out
        self._scale_factor *= factor
        # Clamp
        self._scale_factor = max(0.05, min(self._scale_factor, 40.0))
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta_vec = new_pos - old_pos
        self.translate(delta_vec.x(), delta_vec.y())

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._pixmap_item is not None and self._scale_factor == 1.0:
            self._fit_view(self._pixmap_item.pixmap(), self._pixmap_item)
        else:
            self._center_scene()

    def _fit_view(self, pm: QPixmap, item: QGraphicsPixmapItem) -> None:
        if self.viewport().size().width() <= pm.width() or self.viewport().size().height() <= pm.height():
            self.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.centerOn(item)

    def _center_scene(self) -> None:
        # Center the current scene contents
        self.centerOn(self.sceneRect().center())


class FixedRowHeightDelegate(QStyledItemDelegate):
    def __init__(self, row_height: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._row_height = max(1, int(row_height))

    def sizeHint(self, option, index):  # type: ignore[override]
        sz = super().sizeHint(option, index)
        return QSize(sz.width(), self._row_height)


class SettingsDialog(QDialog):
    def __init__(self, settings: QSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings - PicBed")
        self._settings = settings
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        # S3 settings fields
        self.ed_endpoint = QLineEdit(self)
        self.ed_access = QLineEdit(self)
        self.ed_secret = QLineEdit(self)
        self.ed_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_bucket = QLineEdit(self)
        self.ed_prefix = QLineEdit(self)
        self.ed_region = QLineEdit(self)
        self.ed_sigver = QLineEdit(self)
        self.cb_path_style = QCheckBox("Force Path-Style", self)
        self.cb_unsigned = QCheckBox("Use UNSIGNED-PAYLOAD", self)
        self.ed_upload_tpl = QLineEdit(self)
        self.ed_upload_tpl.setToolTip("Supported placeholders: \n" + "\n".join(SUPPORT_PLACEHOLDERS))
        self.ed_output_pattern = QLineEdit(self)
        self.ed_output_pattern.setToolTip("Supported placeholders: \n" + "\n".join([*OUTPUT_PLACEHOLDERS, *SUPPORT_PLACEHOLDERS]))
        self.cb_over_write_existing = QCheckBox("Overwrite Existing", self)

        # Preview settings
        self.ed_preview_max = QLineEdit(self)
        preview_max_size_validater = QIntValidator(self)
        preview_max_size_validater.setRange(1, 100)    # 100MB
        self.ed_preview_max.setValidator(preview_max_size_validater)

        # Cache settings
        self.ed_cache_max_size = QLineEdit(self)
        cache_max_size_validater = QIntValidator(self)
        cache_max_size_validater.setRange(1, 4 * 1024)    # 4GB
        self.ed_cache_max_size.setValidator(cache_max_size_validater)

        self.ed_cache_max_items = QLineEdit(self)
        cache_max_items_validater = QIntValidator(self)
        cache_max_items_validater.setRange(1, 100)
        self.ed_cache_max_items.setValidator(cache_max_items_validater)

        # Server settings
        self.ed_server_host = QLineEdit(self)
        ip_regex = r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        ipv6_regex = r"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$"
        ip_validator = QRegularExpressionValidator(QRegularExpression(f"^localhost$|{ip_regex}|{ipv6_regex}"))
        self.ed_server_host.setValidator(ip_validator)
        self.ed_server_host.setPlaceholderText("e.g.: 127.0.0.1")
        
        self.ed_server_port = QLineEdit(self)
        self.ed_server_port.setPlaceholderText("e.g.: 36677")
        port_validater = QIntValidator(self)
        port_validater.setRange(1, 65535)
        self.ed_server_port.setValidator(port_validater)
        
        self.ed_server_max_upload_size = QLineEdit(self)
        max_upload_size_validater = QIntValidator(self)
        max_upload_size_validater.setRange(1, 100)
        self.ed_server_max_upload_size.setValidator(max_upload_size_validater)

        # start without mainwindow
        self.cb_start_without_mainwindow = QCheckBox("Hide Main Window On Startup", self)

        form.addRow("Endpoint URL", self.ed_endpoint)
        form.addRow("Access Key", self.ed_access)
        form.addRow("Secret Key", self.ed_secret)
        form.addRow("Bucket", self.ed_bucket)
        form.addRow("Prefix", self.ed_prefix)
        form.addRow("Region", self.ed_region)
        form.addRow("Signature Version", self.ed_sigver)
        form.addRow("Path-Style", self.cb_path_style)
        form.addRow("Unsigned Payload", self.cb_unsigned)
        form.addRow("Upload Path Template", self.ed_upload_tpl)
        form.addRow("Output URL Pattern", self.ed_output_pattern)
        form.addRow("Overwrite Existing", self.cb_over_write_existing)
        form.addRow("Preview Max Size (MB)", self.ed_preview_max)
        form.addRow("Cache Max Size (MB)", self.ed_cache_max_size)
        form.addRow("Cache Max Items", self.ed_cache_max_items)
        form.addRow("Server Host", self.ed_server_host)
        form.addRow("Server Port", self.ed_server_port)
        form.addRow("Server Max Upload Size (MB)", self.ed_server_max_upload_size)
        form.addRow("", self.cb_start_without_mainwindow)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._load_from_settings()

        self.resize(600, self.height())

    def _load_from_settings(self) -> None:
        s = self._settings
        self.ed_endpoint.setText(s.value("s3/endpoint_url", ""))
        self.ed_region.setText(s.value("s3/region_name", ""))
        self.ed_bucket.setText(s.value("s3/bucket", ""))
        self.ed_access.setText(s.value("s3/access_key", ""))
        self.ed_secret.setText(s.value("s3/secret_key", ""))
        self.cb_path_style.setChecked(bool(s.value("s3/use_path_style", True, type=bool)))
        self.ed_prefix.setText(s.value("s3/prefix", ""))
        self.ed_sigver.setText(s.value("s3/signature_version", "s3v4"))
        self.cb_unsigned.setChecked(bool(s.value("s3/unsigned_payload", True, type=bool)))
        self.ed_upload_tpl.setText(s.value("s3/upload_path_template", ""))
        self.ed_output_pattern.setText(s.value("s3/output_url_pattern", ""))
        self.cb_over_write_existing.setChecked(bool(s.value("s3/over_write_existing", False, type=bool)))
        self.ed_preview_max.setText(s.value("preview/max_size_mb", ""))
        self.ed_cache_max_size.setText(s.value("cache/max_size_mb", "100"))
        self.ed_cache_max_items.setText(s.value("cache/max_items", "10"))
        self.ed_server_host.setText(s.value("server/host", "127.0.0.1"))
        self.ed_server_port.setText(s.value("server/port", "36677"))
        self.ed_server_max_upload_size.setText(s.value("server/max_upload_size_mb", "10"))
        self.cb_start_without_mainwindow.setChecked(s.value('picbed/start_without_mainwindow', True, type=bool))

    def save_to_settings(self) -> None:
        s = self._settings
        s.setValue("s3/endpoint_url", self.ed_endpoint.text().strip() or "")
        s.setValue("s3/region_name", self.ed_region.text().strip() or "")
        s.setValue("s3/bucket", self.ed_bucket.text().strip())
        s.setValue("s3/access_key", self.ed_access.text().strip() or "")
        s.setValue("s3/secret_key", self.ed_secret.text().strip() or "")
        s.setValue("s3/use_path_style", self.cb_path_style.isChecked())
        s.setValue("s3/prefix", self.ed_prefix.text().strip() or "")
        s.setValue("s3/signature_version", self.ed_sigver.text().strip() or "s3v4")
        s.setValue("s3/unsigned_payload", self.cb_unsigned.isChecked())
        s.setValue("s3/upload_path_template", self.ed_upload_tpl.text().strip() or "")
        s.setValue("s3/output_url_pattern", self.ed_output_pattern.text().strip() or "")
        s.setValue("s3/over_write_existing", self.cb_over_write_existing.isChecked())
        s.setValue("preview/max_size_mb", self.ed_preview_max.text().strip() or "")
        s.setValue("cache/max_size_mb", self.ed_cache_max_size.text().strip() or "100")
        s.setValue("cache/max_items", self.ed_cache_max_items.text().strip() or "10")
        s.setValue("server/host", self.ed_server_host.text().strip() or "127.0.0.1")
        s.setValue("server/port", self.ed_server_port.text().strip() or "36677")
        s.setValue("server/max_upload_size_mb", self.ed_server_max_upload_size.text().strip() or "10")
        s.setValue("picbed/start_without_mainwindow", self.cb_start_without_mainwindow.isChecked())


class MainWindow(QMainWindow):
    Msg = Signal(str, int)  # message, timeout
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PicBed")
        self.setWindowIcon(QIcon(":/image/picbed.ico"))

        # Task management system(s3 config will setup for TaskManager in `_reload_s3_client_from_settings`)
        self._task_manager = TaskManager()
        self._active_threads: list[WorkerTracer[QThread]] = []
        self._threads_lock = threading.Lock()
        self._resource_monitor = ResourceMonitor()

        # Initialize settings and S3 client from settings
        self._settings = QSettings("picbed", "PicBedApp")
        self.s3: S3Client | None = None
        self._server: PicGoLikeServer | None = None
        self.preview_max_size_bytes: int = -1
        self._reload_s3_client_from_settings(show_error=False)

        self._item_changed_queue: list[str] = []
        self._is_processing_item_changed_queue: bool = False

        # State caches - 使用新的缓存管理器
        self._cache_manager = self._init_cache_manager()
        self._item_by_key: dict[str, QListWidgetItem] = {}
        self._tree_item_by_key: dict[str, QTreeWidgetItem] = {}
        self._obj_by_key: dict[str, dict] = {}
        self._thumb_queue: list[str] = []
        self._thumb_worker: GeneralWorker | None = None
        self._current_preview_pixmap: QPixmap | None = None
        self._current_preview_key: str | None = None
        self._preview_worker: GeneralWorker | PreviewWorker | None = None
        # ETag-aware caches
        self._etag_by_key: dict[str, str] = {}
        self._preview_etag: dict[str, str] = {}
        self._thumb_size = 96
        # Placeholder icon cache by extension (保持简单字典，因为很小)
        self._placeholder_icon_cache: dict[str, QIcon] = {}
        # Max bytes allowed for thumbnail fetching (from env). None means no limit
        self._thumb_max_bytes: int | None = self._read_max_thumb_bytes()

        # new message handler
        self.new_instance_message = win32api.RegisterWindowMessage(APP_KEY)
        # Start local PicGo-like server if S3 is ready
        threading.Thread(target=self._ensure_server_running).start()
        
        # 启动定期缓存清理定时器
        self._cache_cleanup_timer = QTimer(self)
        self._cache_cleanup_timer.timeout.connect(self._cleanup_caches)
        self._cache_cleanup_timer.start(30000)  # 每30秒清理一次
        
        # 启动缓存状态更新定时器
        self._cache_status_timer = QTimer(self)
        self._cache_status_timer.timeout.connect(self._update_cache_status)
        self._cache_status_timer.start(5000)  # 每5秒更新一次状态

        self.init_ui()

        # connect message signal
        self.Msg.connect(self.show_message)

    def init_ui(self) -> None:
        # Root splitter: left (browser) | right (preview)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)
        self.setCentralWidget(self.splitter)

        # Left panel container
        left = QWidget(self)
        v = QVBoxLayout(left)
        v.setContentsMargins(8, 4, 8, 4)
        v.setSpacing(8)

        tophlayout = QHBoxLayout()
        v.addLayout(tophlayout)

        self.prefix_edit = QLineEdit(self)
        self.prefix_edit.setPlaceholderText("Prefix filter (optional)")
        tophlayout.addWidget(self.prefix_edit)

        self.file_count_label = QLabel("0 files", self)
        tophlayout.addWidget(self.file_count_label)

        self.load_all_thumbs_btn = QPushButton("Load All Thumbnails", self)
        tophlayout.addWidget(self.load_all_thumbs_btn)

        # Tree view (alternate view)
        self.tree_widget = QTreeWidget(self)
        self.tree_widget.setHeaderLabels(["", "Key", "LastModified", "Size", "ETag"])
        self.tree_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.tree_widget.setRootIsDecorated(False)
        header = self.tree_widget.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.tree_widget.setIconSize(QSize(64, 64))
        self.tree_widget.setColumnWidth(0, 64)
        # Fix row height to match icon height for consistent spacing
        try:
            self.tree_widget.setUniformRowHeights(True)
        except Exception:
            pass
        self.tree_widget.setItemDelegate(FixedRowHeightDelegate(self.tree_widget.iconSize().height(), self.tree_widget))
        self.tree_widget.setVisible(True)   # 默认detail模式
        self.tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        v.addWidget(self.tree_widget, 1)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_widget.setUniformItemSizes(False)
        self.list_widget.setViewMode(QListView.ViewMode.ListMode)
        self.list_widget.setIconSize(QSize(self._thumb_size, self._thumb_size))
        self.list_widget.setSpacing(8)
        self.list_widget.setVisible(False)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        v.addWidget(self.list_widget, 1)

        h = QHBoxLayout()
        v.addLayout(h)
        self.btn_refresh = QPushButton("Refresh", self)
        self.btn_upload = QPushButton("Upload File(s)", self)
        self.btn_clip = QPushButton("Upload Clipboard", self)
        self.btn_download = QPushButton("Download Selected", self)
        self.btn_delete = QPushButton("Delete Selected", self)
        for b in [self.btn_refresh, self.btn_upload, self.btn_clip, self.btn_download, self.btn_delete]:
            h.addWidget(b)

        # Toolbar for view and actions
        tb = QToolBar("View", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        group = QActionGroup(self)
        group.setExclusive(True)

        self.act_tree = QAction("Detail", self)
        self.act_tree.setCheckable(True)
        self.act_tree.triggered.connect(self._set_tree_mode)
        group.addAction(self.act_tree)
        tb.addAction(self.act_tree)

        self.act_grid = QAction("Grid", self)
        self.act_grid.setCheckable(True)
        self.act_grid.triggered.connect(self._set_grid_mode)
        group.addAction(self.act_grid)
        tb.addAction(self.act_grid)
        # Settings action
        self.act_settings = QAction("Settings", self)
        self.act_settings.triggered.connect(self._open_settings_dialog)
        tb.addAction(self.act_settings)

        # Wire buttons & interactions
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_upload.clicked.connect(self.upload_files)
        self.btn_clip.clicked.connect(self.upload_clipboard)
        self.btn_download.clicked.connect(self.download_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.list_widget.currentItemChanged.connect(self._on_current_item_changed)
        self.tree_widget.currentItemChanged.connect(self._on_current_item_changed)
        self.load_all_thumbs_btn.clicked.connect(self.load_all_thumbs)
        self.list_widget.customContextMenuRequested.connect(lambda p: self._on_context_menu(self.list_widget.mapToGlobal(p), self.list_widget))
        self.tree_widget.customContextMenuRequested.connect(lambda p: self._on_context_menu(self.tree_widget.mapToGlobal(p), self.tree_widget))

        # Status bar
        self._status = self.statusBar()
        self._status.showMessage("Ready")

        # Cache status label in status bar
        self._cache_status_label = QLabel("", self)
        self._cache_status_label.setToolTip("Cache usage: Size/Items")
        self._status.addPermanentWidget(self._cache_status_label)
        
        # Paste (Ctrl+V) to upload clipboard image/file
        self._paste_shortcut = QShortcut(QKeySequence.StandardKey.Paste, self)
        self._paste_shortcut.activated.connect(self.upload_clipboard)

        # Right preview panel
        right = QWidget(self)
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(8, 4, 8, 4)
        right_v.setSpacing(8)
        self.preview_view = ImageGraphicsView(right)
        right_v.addWidget(self.preview_view, 1)
        # File info label
        self.info_label = QLabel(right)
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.info_label.setText("No selection")
        right_v.addWidget(self.info_label)

        # Assemble splitter
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setSizes([300, 300])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        self.refresh()

        # System tray integration
        self._really_quit: bool = False
        self._init_tray()

    # ---------- Cache management ----------
    def _init_cache_manager(self) -> ImageCacheManager:
        """初始化缓存管理器"""
        max_size_mb = 1024
        max_items = 100
        try:
            max_size_mb = float(self._settings.value("cache/max_size_mb", "1024"))
            max_items = int(self._settings.value("cache/max_items", "100"))
        except Exception as e:
            logger.warning(f"Failed to initialize cache manager: {e}, using defaults")
        return ImageCacheManager(max_size_mb, max_items)
    
    def _update_cache_limits(self) -> None:
        """更新缓存限制"""
        try:
            max_size_mb = float(self._settings.value("cache/max_size_mb", "100"))
            max_items = int(self._settings.value("cache/max_items", "10"))
            self._cache_manager.update_limits(max_size_mb, max_items)
            logger.info(f"Updated cache limits: {max_size_mb}MB, {max_items} items")
        except Exception as e:
            logger.error(f"Failed to update cache limits: {e}")
    
    def _cleanup_caches(self) -> None:
        """定期清理缓存"""
        try:
            # 清理超过1小时的缓存项
            removed_count = self._cache_manager.cleanup_all(max_age_seconds=3600)
            if removed_count > 0:
                logger.debug(f"Cleaned up {removed_count} old cache items")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def _update_cache_status(self) -> None:
        """更新缓存状态显示"""
        try:
            stats = self._cache_manager.get_total_stats()
            total_size_mb = stats['total_size_mb']
            max_size_mb = stats['max_size_mb']
            total_items = stats['total_items']
            max_items = stats['max_items']
            
            # 计算使用率
            size_usage = (total_size_mb / max_size_mb * 100) if max_size_mb > 0 else 0
            item_usage = (total_items / max_items * 100) if max_items > 0 else 0
            
            # 更新状态栏显示
            status_text = f"Cache: {total_size_mb:.1f}/{max_size_mb:.0f}MB ({size_usage:.0f}%), {total_items}/{max_items} items ({item_usage:.0f}%)"
            self._cache_status_label.setText(status_text)
            
            # 如果使用率过高，显示警告颜色
            if size_usage > 90 or item_usage > 90:
                self._cache_status_label.setStyleSheet("color: orange;")
            elif size_usage > 80 or item_usage > 80:
                self._cache_status_label.setStyleSheet("color: yellow;")
            else:
                self._cache_status_label.setStyleSheet("")
                
        except Exception as e:
            logger.error(f"Failed to update cache status: {e}")

    # ---------- Settings ----------
    def _ensure_s3_client(self) -> bool:
        if self.s3 is None and (not self._reload_s3_client_from_settings(show_error=False)):
            self.show_message("PicBed Configuration Error: \nPlease configure S3 object storage in settings first", 3000)
            return False
        return True

    def _open_settings_dialog(self) -> None:
        dlg = SettingsDialog(self._settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dlg.save_to_settings()
            # Apply settings
            self._thumb_max_bytes = self._read_max_thumb_bytes()
            self._update_cache_limits()
            self._reload_s3_client_from_settings(show_error=True)
            if self.s3 is not None:
                # Refresh server with new client
                threading.Thread(target=self._ensure_server_running).start()
                self.refresh(show_message=True)

    def _reload_s3_client_from_settings(self, *, show_error: bool):
        try:
            cfg = self._build_s3_config_from_settings()
            if self.s3 is not None and (
                self.s3._config.endpoint_url == cfg.endpoint_url
                and self.s3._config.region_name == cfg.region_name
                and self.s3._config.bucket == cfg.bucket
                and self.s3._config.access_key == cfg.access_key
                and self.s3._config.secret_key == cfg.secret_key
                and self.s3._config.use_path_style == cfg.use_path_style
                and self.s3._config.prefix == cfg.prefix
                and self.s3._config.signature_version == cfg.signature_version
                and self.s3._config.unsigned_payload == cfg.unsigned_payload
                and self.s3._config.remove_sha256_header == cfg.remove_sha256_header
                and self.s3._config.upload_path_template == cfg.upload_path_template
                and self.s3._config.output_url_pattern == cfg.output_url_pattern
                and self.s3._config.over_write_existing == cfg.over_write_existing
                and self.s3._config.max_upload_size == cfg.max_upload_size
            ):
                return True
            if not (cfg.bucket or "").strip():
                self.s3 = None
                return False
            self.s3 = S3Client(cfg)
            # Update server client if running
            if self._server is not None:
                self._server.set_s3_client(self.s3)
            # update task manager with new client
            self._task_manager.setup_s3_config(cfg)
            self._task_manager.s3_pool.clear()
            return True
        except Exception as e:
            self.s3 = None
            logger.exception("Failed to initialize S3: {}", e)
            if show_error:
                QMessageBox.critical(self, "S3 配置错误", str(e))
        return False

    def _ensure_server_running(self) -> None:
        # run picgo like server in background thread
        try:
            if self.s3 is None:
                return
            
            host = self._settings.value("server/host", "127.0.0.1")
            port = int(self._settings.value("server/port", "36677"))
            if self._server is not None and (
                self._server._host != host or self._server._port != port):
                self._server.stop()
                self._server = None
            s3 = S3Client(self.s3._config)
            if self._server is None:
                self._server = PicGoLikeServer(s3, host=host, port=port)
                self._server.path_uploaded.connect(self.on_server_path_uploaded)
                self._server.start()
                self.Msg.emit(f"Local PicBed server is listening on http://{host}:{port}", 3000)
            else:
                self._server.set_s3_client(s3)
                self.Msg.emit(f"Local PicBed server updated, listening on http://{host}:{port}", 3000)
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to start local PicBed server: {e}"
            logger.warning(msg)
            self.Msg.emit(msg, 3000)

    def _build_s3_config_from_settings(self) -> S3Config:
        s = self._settings
        return S3Config(
            endpoint_url=s.value("s3/endpoint_url") or None,
            region_name=s.value("s3/region_name") or None,
            bucket=s.value("s3/bucket", ""),
            access_key=s.value("s3/access_key") or None,
            secret_key=s.value("s3/secret_key") or None,
            use_path_style=bool(s.value("s3/use_path_style", True, type=bool)),
            prefix=s.value("s3/prefix", ""),
            signature_version=s.value("s3/signature_version") or "s3v4",
            unsigned_payload=bool(s.value("s3/unsigned_payload", True, type=bool)),
            remove_sha256_header=bool(s.value("s3/remove_sha256_header", False, type=bool)),
            upload_path_template=s.value("s3/upload_path_template", ""),
            output_url_pattern=s.value("s3/output_url_pattern", ""),
            over_write_existing=bool(s.value("s3/over_write_existing", False, type=bool)),
            max_upload_size=int(s.value("server/max_upload_size_mb", "10") or "10") * 1024 * 1024,
        )

    # ------------------ Task management ------------------
    def _run(self, task_func: Callable[[TaskInfo, ...], Any], task_info: TaskInfo, 
                success_callbacks: list[Callable|None], 
                failed_callbacks: list[Callable[[Exception], None]|None]|None = None):
        """Run a task with the task management system"""
        
        def on_success(result: Any):
            # Call success callbacks
            for callback in success_callbacks or []:
                if callback is None:
                    continue
                try:
                    callback(result)
                except Exception as e:
                    logger.error("Failed to call success callback: {}", e)
            
        def on_failed(e: Exception):
            # Call failed callbacks
            for callback in failed_callbacks or []:
                if callback is None:
                    continue
                try:
                    callback(e)
                except Exception as e:
                    logger.error("Failed to call failed callback: {}", e)

        if failed_callbacks is None or len(failed_callbacks) == 0:
            failed_callbacks = (self._on_error, )

        # Check if task can start immediately
        if self._task_manager.register_task(task_info):
            # Start immediately
            self.th = GeneralWorker(task_func, task_info)
            self.th.setParent(self)
            self.th.result.connect(on_success)
            self.th.failed.connect(on_failed)
            self._track_thread(task_info, self.th)
            self.th.start()
        else:
            # Task was queued, update status
            status = self._task_manager.get_status()
            task_status = status.get(task_info.task_type, {})
            self._status.showMessage(f"{task_info.description} (queued: {task_status.get('pending', 0)}, active: {task_status.get('active', 0)})", 3000)
    
    def _start_queued_task(self, task_info: TaskInfo):
        """Start a queued task (called from task completion)"""
        try:
            if task_info.task_type == TaskType.PREVIEW or task_info.task_type == TaskType.THUMBNAIL:
                logger.info(f"Starting queued task: {task_info.task_id} ({task_info.task_type})")
                self._start_preview_for_task(task_info)
            else:
                # For upload/download tasks, they would be handled differently
                # as they have their own specific parameters
                logger.debug(f"Queued task type {task_info.task_type} not implemented for restart")
        except Exception as e:
            logger.error(f"Failed to start queued task {task_info.task_id}: {e}")

    def _on_error(self, e: Exception) -> None:
        logger.error("Operation failed: {}", e)
        self._status.showMessage(f"Error: {e}", 5000)
        QMessageBox.critical(self, "Error", str(e))

    def refresh(self, *args, show_message: bool = True, select_keys: list[str] | None = None) -> None:
        if not self._ensure_s3_client():
            self._status.showMessage("S3 client not initialized", 3000)
            return
        prefix = self.prefix_edit.text().strip()
        if show_message:
            self._status.showMessage("Refreshing...")
        def task(task_info: TaskInfo):
            if task_info.s3 is None:
                raise Exception("S3 client not initialized")
            return task_info.s3.list_objects(prefix)

        def done(objs: list[dict]):
            self.update_object_list(objs, show_message, select_keys)

        task_info = TaskInfo(
            task_id=f"refresh_{uuid7().hex}",
            task_type=TaskType.DOWNLOAD,  # Refresh involves downloading object list
            description="Refreshing object list",
            priority=TaskPriority.HIGH,
            created_at=time.time(),
        )
        self._run(task, task_info=task_info, success_callbacks=(done,), failed_callbacks=(self._on_error,))

    def update_object_list(self, objs: list[dict], show_message: bool = True, select_keys: list[str] | None = None):
        self._item_by_key.clear()
        self._thumb_queue.clear()
        self.list_widget.clear()
        self.tree_widget.clear()
        self._obj_by_key.clear()
        # Track current set for ETag pruning
        seen_keys: set[str] = set()
        # sort by modified time
        objs.sort(key=lambda x: x.get("LastModified", datetime.datetime.now()), reverse=True)
        for i, obj in enumerate(objs):
            key = obj["Key"]
            seen_keys.add(key)
            self._obj_by_key[key] = obj
            etag = (obj.get("ETag") or "").strip('"') if isinstance(obj.get("ETag"), str) else ""
            # Record latest ETag
            old_etag = self._etag_by_key.get(key)
            self._etag_by_key[key] = etag
            # Invalidate preview cache if ETag changed
            if old_etag is not None and old_etag != etag:
                self._cache_manager.preview_cache.remove(key)
                self._preview_etag.pop(key, None)
                self._cache_manager.thumb_cache.remove(key)
            size = obj.get("Size", 0)
            label = f"{key} ({format_size(size)})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setData(Qt.ItemDataRole.ToolTipRole, label)
            # Placeholder or cached icon for any type; real image thumb loads later
            cached_icon = self._cache_manager.thumb_cache.get(key)
            if cached_icon is not None:
                item.setIcon(cached_icon)
            else:
                item.setIcon(self._placeholder_icon_for_key(key))
            self.list_widget.addItem(item)
            self._item_by_key[key] = item

            # Add to tree view
            dt = obj.get("LastModified")
            if hasattr(dt, "astimezone"):
                try:
                    local_dt = dt.astimezone()
                    last_mod_str = local_dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    last_mod_str = str(dt)
            else:
                last_mod_str = ""
            etag_text = etag
            size_text = format_size(int(size) if isinstance(size, int) else 0)
            tree_item = QTreeWidgetItem(['', key, last_mod_str, size_text, etag_text])
            tree_item.setData(0, Qt.ItemDataRole.UserRole, key)
            tree_item.setData(0, Qt.ItemDataRole.ToolTipRole, label)
            # Show cached icon if available; otherwise type-based placeholder
            cached_icon = self._cache_manager.thumb_cache.get(key)
            if cached_icon is not None:
                pm = cached_icon.pixmap(QSize(self._thumb_size, self._thumb_size))
                if not pm.isNull():
                    tree_item.setIcon(0, QIcon(pm.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)))
            if tree_item.icon(0).isNull():
                tree_item.setIcon(0, self._placeholder_icon_for_key(key))
            self.tree_widget.addTopLevelItem(tree_item)
            self._tree_item_by_key[key] = tree_item

            if (i+1) % 10 == 0:
                QApplication.processEvents()

        # Remove ETag entries for keys that no longer exist
        for stale in list(self._etag_by_key.keys()):
            if stale not in seen_keys:
                self._etag_by_key.pop(stale, None)
                self._cache_manager.preview_cache.remove(stale)
                self._preview_etag.pop(stale, None)
                self._cache_manager.thumb_cache.remove(stale)
                self._item_by_key.pop(stale, None)
                self._tree_item_by_key.pop(stale, None)
        logger.info("Listed {} objects", len(objs))
        if show_message:
            self._status.showMessage(f"Loaded {len(objs)} objects", 3000)
        # Do not prefetch thumbnails by default
        self.file_count_label.setText(f"{len(objs)} files")
        # Optionally select provided keys (e.g., newly uploaded)
        if select_keys:
            try:
                # List view selection
                self.list_widget.clearSelection()
                current_list_item = None
                for k in select_keys:
                    item = self._item_by_key.get(k)
                    if item is not None:
                        item.setSelected(True)
                        current_list_item = item
                if current_list_item is not None:
                    self.list_widget.setCurrentItem(current_list_item)
                    try:
                        self.list_widget.scrollToItem(current_list_item)
                    except Exception:
                        pass
                # Tree view selection
                self.tree_widget.clearSelection()
                current_tree_item = None
                for k in select_keys:
                    titem = self._tree_item_by_key.get(k)
                    if titem is not None:
                        titem.setSelected(True)
                        current_tree_item = titem
                if current_tree_item is not None:
                    self.tree_widget.setCurrentItem(current_tree_item)
                    try:
                        self.tree_widget.scrollToItem(current_tree_item)
                    except Exception:
                        pass
                # Focus whichever view is visible
                if self.tree_widget.isVisible():
                    self.tree_widget.setFocus()
                else:
                    self.list_widget.setFocus()
            except Exception as _e:  # noqa: BLE001
                # Non-fatal if selection fails
                logger.debug("Selection after refresh failed: {}", _e)
        
    # ------------------ Upload / Download / Delete ------------------
    def upload_files(self, task_info: TaskInfo | None = None) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select files to upload")
        if not paths:
            return
        self._start_upload_paths(paths, task_info)

    def upload_clipboard(self) -> None:
        data = read_clipboard_image_bytes()
        if not data:
            QMessageBox.warning(self, "Clipboard", "No image or file in clipboard")
            return
        content, mime = data
        self._start_upload_bytes(content, mime, suggested_name="clipboard.png")

    def _start_upload_paths(self, paths: list[str], task_info: TaskInfo|None=None,
            on_success: Callable[[list[tuple[str, str, str]]], None]|None = None,
            on_failed: Callable[[Exception], None]|None = None,
        ) -> None:
        if not self._ensure_s3_client():
            return
        if not paths:
            return
        total = len(paths)
        self._status.showMessage(f"Uploading {total} file(s) ...")

        def task(task_info: TaskInfo):
            uploaded: list[tuple[str, str, str]] = []  # (key, original_name, local_path)
            for i, p in enumerate(paths):
                orig = os.path.basename(p)
                # If a template is provided, render it; otherwise fallback to auto_rename
                if task_info.s3 is None:
                    raise Exception("S3 client not initialized")
                if task_info.s3.upload_path_template:
                    key = render_placeholders(
                        task_info.s3.upload_path_template,
                        original_name=orig,
                        file_path=p,
                    )
                else:
                    key = auto_rename(orig)
                if task_info.s3 is None:
                    raise Exception("S3 client not initialized")
                task_info.s3.upload_file(p, key, public=True)
                uploaded.append((key, orig, p))
            return uploaded
        
        def done(results: list[tuple[str, str, str]]):
            logger.info("Uploaded {} file(s)", len(results))
            # Build output URLs and copy to clipboard
            urls = [self.s3.public_url_for_output(k, original_name=orig) for (k, orig, _local) in results]
            QGuiApplication.clipboard().setText("\n".join(urls))
            self.show_message(f"Uploaded {len(results)} file(s)", 3000)
            # Pre-fill thumbnail cache using local files for newly uploaded images
            for (k, _orig, local_path) in results:
                try:
                    if is_image_key(k) and os.path.exists(local_path):
                        pm = QPixmap(local_path)
                        if not pm.isNull():
                            self._update_thumbnail_for_key(k, pm)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed to prime thumbnail for {}: {}", k, e)
            keys = [k for (k, _orig, _local) in results]
            self.refresh(show_message=False, select_keys=keys)
        
        task_info = TaskInfo(
            task_id=f"upload_{len(paths)}_{uuid7().hex}",
            task_type=TaskType.UPLOAD,
            description=f"Uploading {total} file(s)",
            priority=TaskPriority.MEDIUM,
            created_at=time.time(),
            s3=S3Client(self.s3._config)
        )
        self._run(task, task_info=task_info, success_callbacks=(on_success, done), failed_callbacks=(on_failed, self._on_error))

    def _start_upload_bytes(self, content: bytes, mime: str, *, suggested_name: str = "image.png", task_info: TaskInfo|None=None) -> None:
        if not self._ensure_s3_client():
            return
        self._status.showMessage("Uploading content...")
        def task(task_info: TaskInfo):
            if task_info.s3.upload_path_template:
                key = render_placeholders(
                    task_info.s3.upload_path_template,
                    original_name=suggested_name,
                    content_bytes=content,
                )
            else:
                key = auto_rename(suggested_name, content)
            task_info.s3.upload_bytes(content, key, content_type=mime, public=True)
            return key
        def done(key):
            logger.info("Uploaded content as {}", key)
            url = task_info.s3.public_url_for_output(key, original_name=suggested_name)
            QGuiApplication.clipboard().setText(url)
            self._status.showMessage(f"Uploaded to {url}", 3000)
            # Pre-fill thumbnail cache using uploaded bytes if it's an image key
            try:
                if is_image_key(key):
                    pm = QPixmap()
                    if pm.loadFromData(content):
                        self._update_thumbnail_for_key(key, pm)
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to prime thumbnail for {}: {}", key, e)
            self.refresh(show_message=False, select_keys=[key])
        
        task_info = TaskInfo(
            task_id=f"upload_bytes_{uuid7().hex}",
            task_type=TaskType.UPLOAD,
            description=f"Uploading {suggested_name}",
            priority=TaskPriority.MEDIUM,
            created_at=time.time(),
        )
        self._run(task, task_info=task_info, success_callbacks=(done,), failed_callbacks=(self._on_error,))

    def on_server_path_uploaded(self, result: dict) -> None:
        if not isinstance(result, dict):
            return
        paths = result.get("result", [])
        message: str = result.get("message", "")
        msg = f"PicBed: {len(paths)} file(s) uploaded. \n{message}".strip()
        self._status.showMessage(msg, 3000)
        if not self.isVisible():
            MessageLabel.info(msg, parent=None, timeout=2000, 
                align=Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignRight, offset=(0, -50), topmost=True
            )
            return
        self.refresh(show_message=False)

    def download_selected(self) -> None:
        if not self._ensure_s3_client():
            return
        keys = self.selected_keys()
        if not keys:
            self._status.showMessage("No selection", 3000)
            return
        target_dir = QFileDialog.getExistingDirectory(self, "Select target folder")
        if not target_dir:
            return
        self._status.showMessage("Downloading...")
        def task(task_info: TaskInfo):
            for k in keys:
                local = Path(target_dir) / Path(k).name
                task_info.s3.download_file(k, str(local))
            return keys
        def done(_):
            logger.info("Downloaded {} file(s) to {}", len(keys), target_dir)
            self._status.showMessage(f"Downloaded to {target_dir}", 4000)
            QMessageBox.information(self, "Downloaded", f"Saved to {target_dir}")
        
        task_info = TaskInfo(
            task_id=f"download_{len(keys)}_{uuid7().hex}",
            task_type=TaskType.DOWNLOAD,
            description=f"Downloading {len(keys)} file(s)",
            priority=TaskPriority.MEDIUM,
            created_at=time.time(),
            s3=S3Client(self.s3._config)
        )
        self._run(task, task_info=task_info, success_callbacks=(done,), failed_callbacks=(self._on_error,))

    def delete_selected(self) -> None:
        if not self._ensure_s3_client():
            return
        keys = self.selected_keys()
        if not keys:
            self._status.showMessage("No selection", 3000)
            return
        if QMessageBox.question(self, "Delete", f"Delete {len(keys)} objects?") != QMessageBox.Yes:
            return
        self._status.showMessage("Deleting...")
        def task(task_info: TaskInfo):
            task_info.s3.delete_objects(keys)
            return keys
        def done(_):
            logger.info("Deleted {} object(s)", len(keys))
            self._status.showMessage("Deleted selected objects", 3000)
            # Clear selection and preview to avoid previewing a deleted object
            self.list_widget.clearSelection()
            self.preview_view.show_message("No selection")
            self._current_preview_pixmap = None
            self.refresh(show_message=False)
        
        task_info = TaskInfo(
            task_id=f"delete_{len(keys)}_{uuid7().hex}",
            task_type=TaskType.DOWNLOAD,  # Delete is similar to download in terms of S3 operations
            description=f"Deleting {len(keys)} object(s)",
            priority=TaskPriority.MEDIUM,
            created_at=time.time(),
            s3=S3Client(self.s3._config)
        )
        self._run(task, task_info=task_info, success_callbacks=(done,), failed_callbacks=(self._on_error,))

    def selected_keys(self) -> list[str]:
        if self.tree_widget.isVisible():
            items = self.tree_widget.selectedItems() or []
            return [i.data(0, Qt.ItemDataRole.UserRole) for i in items]
        elif self.list_widget.isVisible():
            items = self.list_widget.selectedItems() or []
            return [i.data(Qt.ItemDataRole.UserRole) for i in items]
        else:
            return []

    # ------------------ View mode ------------------
    def _set_grid_mode(self) -> None:
        self.tree_widget.setVisible(False)
        self.list_widget.setVisible(True)
        self.list_widget.setViewMode(QListView.ViewMode.IconMode)
        self.list_widget.setWrapping(True)
        self.list_widget.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_widget.setGridSize(QSize(self._thumb_size + 24, self._thumb_size + 24))
        self.list_widget.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        row_step = self._thumb_size // 2
        # Scroll roughly two rows per wheel step
        self.list_widget.verticalScrollBar().setSingleStep(max(10, row_step))

    def _set_tree_mode(self) -> None:
        self.list_widget.setVisible(False)
        self.tree_widget.setVisible(True)

    def _placeholder_icon(self) -> QIcon:
        size = max(32, self._thumb_size)
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        return QIcon(pm)

    def _placeholder_icon_for_key(self, key: str) -> QIcon:
        # Return a system icon based on the file extension; fallback to transparent
        try:
            ext = ""
            if isinstance(key, str):
                ext = Path(key).suffix.lower()
            if ext in self._placeholder_icon_cache:
                return self._placeholder_icon_cache[ext]
            icon = QIcon()
            try:
                provider = QFileIconProvider()
                dummy = f"dummy{ext}" if ext else "dummy"
                icon = provider.icon(QFileInfo(dummy))
            except Exception:
                pass
            if icon.isNull():
                icon = QIcon.fromTheme("image-x-generic") or QIcon.fromTheme("image")
            if icon.isNull():
                icon = self._placeholder_icon()
            self._placeholder_icon_cache[ext] = icon
            return icon
        except Exception:
            return self._placeholder_icon()

    # ------------------ Context menu ------------------
    def _on_context_menu(self, global_pos: QPoint, sender: QListWidget|QTreeWidget) -> None:
        if not sender.isVisible():
            return
        menu_style = '''
            QMenu::item {
                padding: 6px 20px;
                icon-size: 0px;             /* 图标大小. 必须padding和icon-size一起设置才会生效 */
            }
            QMenu::item:hover,QMenu::item:selected {
                background-color: #334CC2FF;   /* 悬停/选中时的背景色, AARRGGBB */
                border: none;
                border-radius: 6px;
            }
        '''
        menu = QMenu(self)        
        menu.setFont(QFont("Microsoft YaHei", 11))
        menu.setStyleSheet(menu_style)
        act_copy = menu.addAction("Copy URL(s)")

        # if any item under this pos, show `open in browser`
        pos = sender.mapFromGlobal(global_pos)
        item = sender.itemAt(pos)
        act_open_in_browser = None
        if item is not None:
            # get url for this item
            key = item.data(Qt.ItemDataRole.UserRole) if isinstance(item, QListWidgetItem) else item.data(0, Qt.ItemDataRole.UserRole)
            url = self.s3.public_url_for_output(key) if self.s3 is not None else None
            if url is not None:
                menu.addSeparator()
                act_open_in_browser = menu.addAction(f"Open in Browser")

        action = menu.exec(global_pos)
        if action == act_copy:
            self._copy_selected_urls()
        elif action == act_open_in_browser:
            self._open_in_browser(url)

    def _copy_selected_urls(self) -> None:
        keys = self.selected_keys()
        if not keys:
            self._status.showMessage("No selection", 3000)
            return
        try:
            urls = [self.s3.public_url_for_output(k) for k in keys]
            QGuiApplication.clipboard().setText("\n".join(urls))
            self._status.showMessage(f"Copied {len(keys)} URL(s)", 3000)
        except Exception as e:  # noqa: BLE001
            self._on_error(e)

    def _open_in_browser(self, url: str) -> None:
        QDesktopServices.openUrl(QUrl(url))

    # ------------------ Size limit ------------------
    def _read_max_thumb_bytes(self) -> int | None:
        raw = self._settings.value("preview/max_size_mb", "") or os.getenv("PREVIEW_MAX_SIZE") or ""
        s = raw.strip().lower()
        try:
            if s.endswith("kb"):
                return int(float(s[:-2].strip()) * 1024)
            elif s.endswith("mb"):
                return int(float(s[:-2].strip()) * 1024 * 1024)
            elif s.endswith("gb"):
                return int(float(s[:-2].strip()) * 1024 * 1024 * 1024)
            elif s.endswith("k"):
                return int(float(s[:-1].strip()) * 1024)
            elif s.endswith("m"):
                return int(float(s[:-1].strip()) * 1024 * 1024)
            elif s.endswith("g"):
                return int(float(s[:-1].strip()) * 1024 * 1024 * 1024)
            else:
                # default to MB
                return int(float(s) * 1024 * 1024)
        except Exception:
            return None

    def _is_too_large_for_thumbnail(self, key: str) -> tuple[bool, str]:
        limit = self._thumb_max_bytes
        if not limit:
            return (False, "")
        obj = self._obj_by_key.get(key) or {}
        size_val = obj.get("Size")
        if isinstance(size_val, int) and size_val > int(limit):
            from picbed.utils import format_size
            return (True, f"{format_size(size_val)} > limit {format_size(limit)}")
        return (False, "")

    # ------------------ Thumbnails ------------------
    def load_all_thumbs(self) -> None:
        if not self._ensure_s3_client():
            return
        for key in self._etag_by_key.keys():
            if self._cache_manager.thumb_cache.get(key) is not None:
                continue

            # check if key is in blurry cache
            blurry_pm = self._cache_manager.blurry_cache.get(key)
            if blurry_pm is not None:
                self._update_thumbnail_for_key(key, blurry_pm)
                continue
            
            # check if key is in preview cache
            pvd = self._cache_manager.preview_cache.get(key)
            if pvd is not None and pvd.pixmap is not None:
                self._update_thumbnail_for_key(key, pvd.pixmap)
                continue

            # if any thread is loading current key
            if any(wt.task_info.obj_key == key and wt.task_info.task_type == TaskType.PREVIEW for wt in self._active_threads):
                continue
            
            # Enforce size limit before downloading
            too_large, msg = self._is_too_large_for_thumbnail(key)
            if too_large:
                # Indicate on status bar once, and set a small placeholder
                if msg:
                    self._status.showMessage(msg, 4000)
                continue

            # Use task management system for preview
            task_info = TaskInfo(
                task_id=f"preview_{key}_{uuid7().hex}",
                task_type=TaskType.PREVIEW,
                obj_key=key,
                description=f"Loading preview for {key}",
                priority=TaskPriority.LOW,
                created_at=time.time(),
            )
            
            # Check if task can start immediately
            if self._task_manager.register_task(task_info):
                self._start_preview_for_task(task_info=task_info)

    def _update_thumbnail_for_key(self, key: str, pm: QPixmap):
        if pm is None or pm.isNull():
            return
        icon = QIcon(pm.scaled(self._thumb_size, self._thumb_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        # 估算图标大小并存储到缓存管理器
        icon_size = estimate_qicon_size(icon)
        self._cache_manager.thumb_cache.put(key, icon, icon_size)
        item = self._item_by_key.get(key)
        if item is not None:
            item.setIcon(icon)
        titem = self._tree_item_by_key.get(key)
        if titem is not None:
            titem.setIcon(0, icon)
        self.list_widget.viewport().update()

    # ------------------ Preview ------------------
    def _on_current_item_changed(self, current: QListWidgetItem | QTreeWidgetItem | None, _prev: QListWidgetItem | QTreeWidgetItem | None) -> None:
        key = (current.data(Qt.ItemDataRole.UserRole) if isinstance(current, QListWidgetItem) 
            else current.data(0, Qt.ItemDataRole.UserRole) if isinstance(current, QTreeWidgetItem) 
            else None)
        if not key:
            self.preview_view.show_message("No selection")
            self._current_preview_pixmap = None
            self._current_preview_key = None
            self.info_label.setText("No selection")
            return       
        self._current_preview_key = key

        self._item_changed_queue.append(key)
        QTimer.singleShot(10, self._process_item_changed_queue)

    def _process_item_changed_queue(self) -> None:
        if not self._item_changed_queue:
            return
        if self._is_processing_item_changed_queue:
            return
        self._is_processing_item_changed_queue = True
        try:
            while self._item_changed_queue:
                key = self._item_changed_queue.pop(0)
                self._process_item_changed(key)
        finally:
            self._is_processing_item_changed_queue = False

    def _process_item_changed(self, key: str) -> None:
        # check if selected is image
        if not is_image_key(key):
            self.preview_view.show_message("Not an image")
            self._current_preview_pixmap = None
            self._current_preview_key = None
            self.info_label.setText("Not an image")
            return

        is_too_large, msg = self._is_too_large_for_thumbnail(key)
        if is_too_large:
            self.preview_view.show_message(msg)
            self._current_preview_pixmap = None
            self._current_preview_key = None
            self.info_label.setText(msg)
            return
        
        # check if key is in preview cache
        pvd = self._cache_manager.preview_cache.get(key)
        if pvd is not None:
            if pvd.is_animated:
                self._current_preview_pixmap = None
                self.preview_view.set_movie(pvd.data)
                self._update_info_panel(key, pixel_size=pvd.pixmap.size())
            else:
                self._current_preview_pixmap = pvd.pixmap
                self.preview_view.set_pixmap(pvd.pixmap)
                self._update_info_panel(key, pixel_size=pvd.pixmap.size())
            self._status.clearMessage()
            return

        # check if key is in blurry cache
        blurry_pm = self._cache_manager.blurry_cache.get(key)
        if blurry_pm is not None:
            self._current_preview_pixmap = blurry_pm
            self.preview_view.set_pixmap(blurry_pm)
            self._update_info_panel(key, pixel_size=blurry_pm.size())
            self._status.clearMessage()
        
        # check if key is already in preview worker
        with self._threads_lock:
            if any(wt.task_info.obj_key == key and wt.task_info.task_type == TaskType.PREVIEW for wt in self._active_threads):
                return

        self._status.showMessage("Downloading preview...")
        self.preview_view.show_message('Downloading preview...')

        # Use task management system for preview
        task_info = TaskInfo(
            task_id=f"preview_{key}_{uuid7().hex}",
            task_type=TaskType.PREVIEW,
            obj_key=key,
            description=f"Loading preview for {key}",
            priority=TaskPriority.HIGH,
            created_at=time.time(),
        )
        
        # Check if task can start immediately
        if self._task_manager.register_task(task_info):
            self._start_preview_for_task(task_info=task_info)
        else:
            # Preview task was queued, show message
            status = self._task_manager.get_status()
            preview_status = status.get(TaskType.PREVIEW, {})
            self._status.showMessage(f"Preview queued (active: {preview_status.get('active', 0)}, pending: {preview_status.get('pending', 0)})", 3000)
            self.preview_view.show_message(f"Preview queued (active: {preview_status.get('active', 0)}, pending: {preview_status.get('pending', 0)})")

    def _start_preview_for_task(self, task_info: TaskInfo) -> bool:
        """Start a preview task for the given task info"""
        if not task_info.s3:
            return False

        # check if key is already in preview cache
        pvd = self._cache_manager.preview_cache.get(task_info.obj_key)
        if pvd is not None:
            self._update_thumbnail_for_key(task_info.obj_key, pvd.pixmap)
            next_task = self._task_manager.complete_task(task_info.task_id, task_info.task_type)
            if next_task is not None:
                # use timer to avoid recursive call
                QTimer.singleShot(10, lambda: self._start_preview_for_task(next_task))
            return True
        
        # check if key is already in preview worker
        with self._threads_lock:
            if any(wt.task_info.obj_key == task_info.obj_key and wt.task_info.task_type == TaskType.PREVIEW for wt in self._active_threads):
                return True
        
        # Create and start preview worker
        self._preview_worker = PreviewWorker(task_info.obj_key, task_info.s3, self.preview_view.viewport().size())
        self._preview_worker.setParent(self)
        
        # Set up the same callbacks as in _on_current_item_changed
        def progress(state: str, ret_key: str, p: float) -> None:
            if ret_key != self._current_preview_key:
                return
            msg = f"Downloading image... {p:.2%}" if p < 1 else "Loading image..."    
            self._status.showMessage(msg)
            self.preview_view.show_message(msg)

        def blurry(ret_key: str, pm_or_img: object) -> None:
            if isinstance(pm_or_img, QPixmap):
                pm = pm_or_img
            else:
                pm = QPixmap.fromImage(pm_or_img)  # type: ignore[arg-type]

            # 存储到模糊预览缓存
            pm_size = estimate_qpixmap_size(pm)
            self._cache_manager.blurry_cache.put(ret_key, pm, pm_size)
            
            # load to thumb cache (theoretically, it should NOT be in the cache while loading blurry preview)
            self._update_thumbnail_for_key(ret_key, pm)
            
            if ret_key != self._current_preview_key:
                return
            self._current_preview_pixmap = pm
            self.preview_view.set_pixmap(pm)
            self._status.showMessage("Loading preview...")

        def done(pvd: PreviewData):
            self._status.showMessage("Ready", 2000)
            if pvd.is_error and pvd.key == self._current_preview_key:
                if isinstance(pvd.data, HTTPError) and getattr(pvd.data, "code", None) == 404:
                    self.preview_view.show_message(f"Object not found (deleted): {pvd.data}")
                    self._current_preview_pixmap = None
                    self._status.showMessage("Object not found", 3000)
                    self._update_info_panel(str(pvd.key))
                    return
                self._on_error(f"Error: {str(pvd.data)[:300]}")
                return
            
            # cache preview and clear blurry cache 
            self._cache_manager.blurry_cache.remove(pvd.key)

            if pvd.is_animated:
                pvd: PreviewData = pvd
                if pvd.pixmap is not None:
                    pm_first = pvd.pixmap
                    if not pm_first.isNull():
                        self._update_thumbnail_for_key(pvd.key, pm_first)
                        self._update_info_panel(str(pvd.key), pixel_size=pm_first.size(), animated=True)
                # 存储动画预览到缓存
                preview_size = estimate_qpixmap_size(pvd.pixmap)
                self._cache_manager.preview_cache.put(pvd.key, pvd, preview_size)
                self._current_preview_pixmap = None
                self.preview_view.set_movie(pvd.data)
                return
            
            # still image; convert payload (QImage or QPixmap) on GUI thread
            preview_size = estimate_qpixmap_size(pvd.pixmap)
            self._cache_manager.preview_cache.put(pvd.key, pvd, preview_size)
            self._preview_etag[pvd.key] = self._etag_by_key.get(pvd.key) or ""

            # load to thumb cache (if not in the cache. 
            # If the image is too small, load blurry preview will be ignored
            # that why we need to check if it is in the cache)
            if self._cache_manager.thumb_cache.get(pvd.key) is None:
                self._update_thumbnail_for_key(pvd.key, pvd.pixmap)
            
            if pvd.key != self._current_preview_key:
                # if key is not the current preview key, skip
                return
            
            # update preview view
            self._current_preview_pixmap = pvd.pixmap
            self.preview_view.set_pixmap(pvd.pixmap, update=True)
            self._update_info_panel(str(pvd.key), pixel_size=pvd.pixmap.size())
        
        self._preview_worker.progress.connect(progress)
        self._preview_worker.blurry.connect(blurry)
        self._preview_worker.result.connect(done)
        self._preview_worker.failed.connect(self._on_error)
        self._track_thread(task_info, self._preview_worker)
        self._preview_worker.start()
        return True
    
    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_preview_scaled()

    def _update_preview_scaled(self) -> None:
        # Keep fit behavior on resize when needed
        if self._current_preview_pixmap is None:
            return
        self.preview_view.fit_if_needed()

    # ------------------ Info panel ------------------
    def _update_info_panel(self, key: str, *, pixel_size: QSize | None = None, animated: bool | None = None) -> None:
        obj = self._obj_by_key.get(key) or {}
        size_val = obj.get("Size")
        size_text = format_size(int(size_val)) if isinstance(size_val, (int, float)) else (str(size_val or ""))
        etag = obj.get("ETag") or ""
        if isinstance(etag, str):
            etag = etag.strip('"')
        dt = obj.get("LastModified")
        if hasattr(dt, "astimezone"):
            try:
                dt_local = dt.astimezone()
                dt_text = dt_local.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                dt_text = str(dt)
        else:
            dt_text = str(dt or "")
        mime = obj.get("ContentType") or mimetypes.guess_type(key)[0] or "unknown"
        dims = ""
        if pixel_size is not None:
            dims = f"{pixel_size.width()} x {pixel_size.height()}"
            if animated:
                dims += " (animated)"
        lines = [
            f"Key: {key}",
            f"Size: {size_text}",
            f"MIME: {mime}",
            f"ETag: {etag}",
            f"Modified: {dt_text}",
        ]
        if dims:
            lines.append(f"Dimensions: {dims}")
        else:
            lines.append("")
        self.info_label.setText("\n".join(lines))

    # ------------------ Drag & Drop ------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        mime = event.mimeData()
        if mime is None:
            return
        if mime.hasUrls() or mime.hasImage():
            event.acceptProposedAction()
            return True
        return super().dragEnterEvent(event)
    
    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        mime = event.mimeData()
        if mime is None:
            return
        if mime.hasUrls():
            locals: list[str] = []
            for u in mime.urls():
                local = u.toLocalFile()
                if local:
                    locals.append(local)
            if locals:
                self._start_upload_paths(locals)
        if mime.hasImage():
            image = mime.imageData()
            if isinstance(image, QImage) and not image.isNull():
                byte_array = QByteArray()
                buffer = QBuffer(byte_array)
                buffer.open(QBuffer.OpenModeFlag.WriteOnly)
                image.save(buffer, "PNG")
                buffer.close()  
                self._start_upload_bytes(bytes(byte_array), "image/png", suggested_name="dropped.png")
        return super().dropEvent(event)

    # ------------------ Thread lifecycle ------------------
    def _track_thread(self, task_info: TaskInfo, t: QThread) -> None:
        if not task_info.is_registered:
            logger.warning(f"Task {task_info.task_id} is not registered, please make sure to register the task before tracking it")
        task_info.started_at = time.time()
        wt = WorkerTracer(task_info, t)
        with acquire_lock(self._threads_lock, timeout=1.0) as acquired:
            if not acquired:
                logger.warning("Failed to acquire _threads_lock; thread tracking may be delayed")
            else:
                self._active_threads.append(wt)
        wt._worker.finished.connect(lambda *args, tid=wt.task_info.task_id: self._remove_thread(tid))

    def _remove_thread(self, tid: str):
        wt: WorkerTracer[QThread]|None = None
        with acquire_lock(self._threads_lock, timeout=1.0) as acquired:
            try:
                if acquired:
                    wt = next(wt for wt in self._active_threads if wt.task_info.task_id == tid)
                    wt.task_info.finished_at = time.time()
                    self._active_threads.remove(wt)
                    wt._worker.deleteLater()
            except StopIteration:
                pass
            except Exception as e:
                logger.error(f"Error removing thread: {e}")
        if wt is not None:
            next_task = self._task_manager.complete_task(wt.task_info.task_id, wt.task_info.task_type)
            if next_task is not None:
                self._start_queued_task(next_task)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        # If user closes the window, minimize to tray unless we are quitting intentionally
        try:
            if not self._really_quit:
                event.ignore()
                self.hide()
                if not self.tray_infoed:
                    self.tray_infoed = True
                    try:
                        self.tray_icon.showMessage("", "PicBed 已最小化到系统托盘。双击图标可恢复窗口。")
                    except Exception:
                        pass
                return
        except Exception:
            # Fall through to normal close path
            pass

        # Shutdown task manager to prevent new tasks
        self._task_manager.shutdown()
        
        # Stop timers
        if hasattr(self, '_cache_cleanup_timer'):
            self._cache_cleanup_timer.stop()
        if hasattr(self, '_cache_status_timer'):
            self._cache_status_timer.stop()
        
        # Clear all caches
        if hasattr(self, '_cache_manager'):
            self._cache_manager.clear_all()
        
        # Prevent starting new tasks
        self._thumb_queue.clear()
        
        # Try to stop/finish background threads with thread safety
        with acquire_lock(self._threads_lock, timeout=1.0) as acquired:
            if acquired:
                active_threads = list(self._active_threads)
            else:
                active_threads = []
        
        for wt in active_threads:
            try:
                wt._worker.requestInterruption()  # may not be honored by our tasks
            except Exception:
                pass
        
        # Wait a bit for clean finish
        for wt in active_threads:
            try:
                wt._worker.wait(1500)
            except Exception:
                pass
        
        # Force terminate any stubborn ones to avoid QThread destruction warning on exit
        for wt in active_threads:
            if wt._worker.isRunning():
                try:
                    wt._worker.terminate()
                    wt._worker.wait(200)
                except Exception:
                    pass
        # Stop local server
        try:
            if self._server is not None:
                self._server.stop()
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------ System Tray ------------------
    def _init_tray(self) -> None:
        try:
            if not QSystemTrayIcon.isSystemTrayAvailable():
                return
        except Exception:
            return
        menu_style = '''
            QMenu::item:hover,QMenu::item:selected {
                background-color: #334CC2FF;   /* 悬停/选中时的背景色, AARRGGBB */
                border: none;
                border-radius: 6px;
            }
        '''
        icon = QIcon(":/image/picbed.ico")
        self.tray_icon = QSystemTrayIcon(icon, self)
        self.tray_infoed = False
        self.tray_icon.setToolTip("PicBed")

        tray_menu = QMenu(self)
        tray_menu.setFont(QFont("Microsoft YaHei", 11))
        tray_menu.setStyleSheet(menu_style)

        act_show = tray_menu.addAction("显示主窗口")
        act_show.setIcon(QIcon(":/image/picbed.ico"))

        act_clip = tray_menu.addAction("上传剪贴板")
        act_clip.setIcon(QIcon(":/image/clipboard-96.png"))

        act_upload = tray_menu.addAction("上传文件")
        act_upload.setIcon(QIcon(":/image/upload.png"))

        act_clear_cache = tray_menu.addAction("清空缓存")
        act_clear_cache.setIcon(QIcon(":/image/clear-cache.png"))

        tray_menu.addSeparator()

        act_settings = tray_menu.addAction("设置")
        act_settings.setIcon(QIcon(":/image/settings.png"))
        tray_menu.addSeparator()

        act_restart = tray_menu.addAction("重启")
        act_restart.setIcon(QIcon(":/image/restart.png"))
        
        act_quit = tray_menu.addAction("退出")
        act_quit.setIcon(QIcon(":/image/exit.png"))

        act_show.triggered.connect(self._show_main_window)
        act_settings.triggered.connect(self._open_settings_dialog)
        act_clear_cache.triggered.connect(self._clear_cache)
        act_clip.triggered.connect(self.upload_clipboard)
        act_upload.triggered.connect(self.upload_files)
        act_quit.triggered.connect(self._quit_app)
        act_restart.triggered.connect(restart_app)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        try:
            self.tray_icon.show()
        except Exception:
            pass

    def _on_tray_activated(self, reason):
        try:
            if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
                self._show_main_window()
        except Exception:
            self._show_main_window()

    def _quit_app(self) -> None:
        self._really_quit = True
        self.tray_icon.hide()
        try:
            self.close()
            QApplication.quit()
        except Exception:
            pass

    def _clear_cache(self) -> None:
        self._cache_manager.clear_all()
        self.show_message("缓存已清空")

    # ------------------ Native events ------------------
    def nativeEvent(self, eventType, message):
        """
        处理Windows原生事件，包括新实例消息和WM_COPYDATA
        """
        try:
            if sys.platform == 'win32' and eventType == b'windows_generic_MSG':
                msg = wintypes.MSG.from_address(message.__int__())
                if msg.message == self.new_instance_message:
                    self.handle_new_instance_message()
                    return True, 0
        except Exception as e:
            logger.error(f"处理原生事件失败: {e}")
        return super().nativeEvent(eventType, message)

    def handle_new_instance_message(self) -> None:
        # parse startup args
        tempfile = temp_dir() / ARGS_TEMP_PKL_FILE_NAME
        if not tempfile.exists():
            self._show_main_window()
            return
        
        try:
            # 读取命令行参数并执行
            with open(tempfile, "rb") as f:
                args: Namespace = pickle.load(f)
            tempfile.unlink()
        except Exception as e:
            logger.error(f"读取命令行参数失败: {e}")
            self.show_message(f"读取命令行参数失败: {e}", 3000)
            self._show_main_window()
            return
        
        if not isinstance(args, Namespace) or 'command' not in args:
            self._show_main_window()
            return

        if process_cli_args(self, args):
            return

        # if not handled, show main window
        self._show_main_window()

    def _show_main_window(self) -> None:
        self.show()
        self.activateWindow()
        self.raise_()

    # ------------------ Message ------------------
    def show_message(self, message: str, timeout: int = 3000) -> None:
        self._status.showMessage(message, timeout)
        if not self.isVisible():
            MessageLabel.info(message, parent=None, timeout=timeout, 
                align=Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignRight, offset=(0, -50),
                processEvent=True, topmost=True
            )


def process_cli_args(w: MainWindow, args: Namespace) -> bool:
    def on_upload_success(results: list[tuple[str, str, str]]):
        ''' 上传成功回调 '''
        MessageLabel.info(f"上传了 {len(results)} 个文件, 上传结果已复制到剪贴板", parent=None, timeout=2000, 
            align=Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignRight, offset=(0, -50),
            processEvent=True, topmost=True
        )
        QApplication.processEvents()
    
    def on_upload_failed(e: Exception):
        ''' 上传失败回调 '''
        MessageLabel.error(f"上传失败: {e}", parent=None, timeout=2000, 
            align=Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignRight, offset=(0, -50),
            processEvent=True, topmost=True
        )
        QApplication.processEvents()
    
    if args.command == "upload":
        paths = args.files
        w._start_upload_paths(paths, on_success=on_upload_success, on_failed=on_upload_failed)
        return True

    return False


def restart_app() -> None:
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main() -> None:
    # Configure logging early
    args = parse_startup_args()
    logdir = appdir() / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    if sys.stderr is not None and sys.stderr.isatty():
        logger.add(sys.stderr, level=args.logcli_level, enqueue=True, colorize=True, backtrace=True, diagnose=True)
    logger.add(logdir / "picbed.log", rotation="10 MB", retention=10, level=args.logfile_level, enqueue=True)

    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.resize(1000, 600)

    if w._settings.value('picbed/start_without_mainwindow', True, type=bool):
        w.hide()
        w.tray_infoed = True
        MessageLabel.info("PicBed 已最小化到系统托盘。双击图标可恢复窗口。", parent=None, timeout=2000,
            align=Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignRight, offset=(0, -50),
            processEvent=True, topmost=True
        )
    else:
        w.show()

    set_window_handle(get_shm(), w.winId())

    QApplication.processEvents()

    process_cli_args(w, args)

    app.exec()

