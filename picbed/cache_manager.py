"""
缓存管理器模块

提供带大小限制的LRU缓存管理功能，支持多种缓存类型。
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Generic, Optional, TypeVar, Union

from loguru import logger
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QPixmap
from pympler import asizeof

from picbed.utils import format_size


@dataclass
class PreviewData:
    key: str
    data: bytes
    pixmap: QPixmap | None = None
    size: QSize|None = None
    icon: QIcon|None = None
    is_animated: bool = False
    is_error: bool = False



_DATA = TypeVar('DATA', bound=Union[PreviewData])

class CacheItem(Generic[_DATA]):
    """缓存项，包含数据和访问时间"""
    
    def __init__(self, data: _DATA, size_bytes: int = 0):
        self.data = data
        self.size_bytes = size_bytes
        self.access_time = time.time()
        self.access_count = 1
    
    def touch(self):
        """更新访问时间和计数"""
        self.access_time = time.time()
        self.access_count += 1


class CacheManager(Generic[_DATA]):
    """带大小限制的LRU缓存管理器"""
    
    def __init__(self, name: str, max_size_mb: float = 1000, max_items: int = 100):
        """
        初始化缓存管理器
        
        Args:
            max_size_mb: 最大缓存大小（MB）
            max_items: 最大缓存项数量
        """
        self._name = name
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_items = max_items
        self.current_size_bytes = 0
        self.current_items = 0
        
        # 使用OrderedDict实现LRU
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        
        # 统计信息
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"CacheManager initialized: name={name}, max_size={max_size_mb}MB, max_items={max_items}")
    
    def get(self, key: str) -> Optional[_DATA]:
        """获取缓存项"""
        if key in self._cache:
            # 移动到末尾（最近使用）
            item = self._cache.pop(key)
            item.touch()
            self._cache[key] = item
            self.hits += 1
            return item.data
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, data: _DATA, size_bytes: int = 0) -> None:
        """添加或更新缓存项"""
        # 如果已存在，先移除
        if key in self._cache:
            old_item = self._cache.pop(key)
            self.current_size_bytes -= old_item.size_bytes
            self.current_items -= 1
        
        # 创建新项
        item = CacheItem(data, size_bytes)
        self._cache[key] = item
        self.current_size_bytes += size_bytes
        self.current_items += 1
        
        # 检查是否需要清理
        self._cleanup_if_needed()
    
    def remove(self, key: str) -> bool:
        """移除缓存项"""
        if key in self._cache:
            item = self._cache.pop(key)
            self.current_size_bytes -= item.size_bytes
            self.current_items -= 1
            return True
        return False
    
    def clear(self) -> None:
        """清空所有缓存"""
        self._cache.clear()
        msg = f"{self._name} cleared, removed {self.current_items} items, {format_size(self.current_size_bytes)}"
        self.current_size_bytes = 0
        self.current_items = 0
        logger.info(msg)
    
    def _cleanup_if_needed(self) -> None:
        """根据需要清理缓存"""
        # 按大小清理
        while ((self.max_size_bytes > 0 and self.current_size_bytes > self.max_size_bytes) or 
               (self.max_items > 0 and self.current_items > self.max_items)) and self._cache:
            # 移除最久未使用的项
            key, item = self._cache.popitem(last=False)
            self.current_size_bytes -= item.size_bytes
            self.current_items -= 1
            self.evictions += 1
            logger.debug(f"{self._name} evicted cache item: {key} (size: {format_size(item.size_bytes)})")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """获取缓存统计信息"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        return {
            'current_size_mb': self.current_size_bytes / (1024 * 1024) if self.max_size_bytes > 0 else 0,
            'max_size_mb': self.max_size_bytes / (1024 * 1024) if self.max_size_bytes > 0 else 0,
            'current_items': self.current_items if self.max_items > 0 else 0,
            'max_items': self.max_items if self.max_items > 0 else 0,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }
    
    def cleanup_old_items(self, max_age_seconds: float = 3600) -> int:
        """清理超过指定时间的缓存项"""
        current_time = time.time()
        removed_count = 0
        
        # 从最旧的开始检查
        keys_to_remove = []
        for key, item in self._cache.items():
            if current_time - item.access_time > max_age_seconds:
                keys_to_remove.append(key)
            else:
                # 由于是OrderedDict，一旦遇到未过期的项就可以停止
                break
        
        for key in keys_to_remove:
            item = self._cache.pop(key)
            self.current_size_bytes -= item.size_bytes
            self.current_items -= 1
            removed_count += 1
        
        if removed_count > 0:
            logger.info(f"{self._name} cleaned up {removed_count} old cache items, {format_size(self.current_size_bytes)}")
        else:
            logger.debug(f"{self._name} no old cache items to clean up")
        
        return removed_count


class ImageCacheManager:
    """专门用于图片缓存的管理器"""
    
    def __init__(self, max_size_mb: float = 100.0, max_items: int = 1000):
        self.thumb_cache = CacheManager[QIcon]("thumb cache", -1, -1)
        self.preview_cache = CacheManager[PreviewData]("preview cache", max_size_mb, max_items)
        self.blurry_cache = CacheManager[QPixmap]("blurry cache", -1, -1)
        
        self.max_size_mb = max_size_mb
        self.max_items = max_items
    
    def update_limits(self, max_size_mb: float, max_items: int) -> None:
        """更新缓存限制"""
        self.max_size_mb = max_size_mb
        self.max_items = max_items
        
        self.preview_cache.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.preview_cache.max_items = max_items

        # icon无需限制
        self.thumb_cache.max_size_bytes = -1
        self.thumb_cache.max_items = -1

        # 模糊预览会在高清预览加载时被清理
        self.blurry_cache.max_size_bytes = -1
        self.blurry_cache.max_items = -1
        
        logger.info(f"Updated cache limits: {self.preview_cache._name} {max_size_mb}MB, {max_items} items")

        self.thumb_cache._cleanup_if_needed()
        self.preview_cache._cleanup_if_needed()
        self.blurry_cache._cleanup_if_needed()
    
    def clear_all(self) -> None:
        """清空所有缓存"""
        self.thumb_cache.clear()
        self.preview_cache.clear()
        self.blurry_cache.clear()
    
    def cleanup_all(self, max_age_seconds: float = 3600) -> int:
        """清理所有缓存中的过期项"""
        total_removed = 0
        total_removed += self.thumb_cache.cleanup_old_items(max_age_seconds)
        total_removed += self.preview_cache.cleanup_old_items(max_age_seconds)
        total_removed += self.blurry_cache.cleanup_old_items(max_age_seconds)
        return total_removed
    
    def get_total_stats(self) -> Dict[str, Union[int, float]]:
        """获取所有缓存的统计信息"""
        thumb_stats = self.thumb_cache.get_stats()
        preview_stats = self.preview_cache.get_stats()
        blurry_stats = self.blurry_cache.get_stats()
        
        return {
            'total_size_mb': (thumb_stats['current_size_mb'] + 
                             preview_stats['current_size_mb'] + 
                             blurry_stats['current_size_mb']),
            'max_size_mb': self.max_size_mb,
            'total_items': (thumb_stats['current_items'] + 
                           preview_stats['current_items'] + 
                           blurry_stats['current_items']),
            'max_items': self.max_items,
            'total_hits': thumb_stats['hits'] + preview_stats['hits'] + blurry_stats['hits'],
            'total_misses': thumb_stats['misses'] + preview_stats['misses'] + blurry_stats['misses'],
            'total_evictions': (thumb_stats['evictions'] + 
                               preview_stats['evictions'] + 
                               blurry_stats['evictions']),
            'thumb_cache': thumb_stats,
            'preview_cache': preview_stats,
            'blurry_cache': blurry_stats
        }


def estimate_qpixmap_size(qpixmap: QPixmap) -> int:
    """估算QPixmap的内存占用大小"""
    try:
        return qpixmap.width() * qpixmap.height() * qpixmap.depth() // 8 + asizeof.asizeof(qpixmap)
    except Exception:
        pass
    return asizeof.asizeof(qpixmap)


def estimate_qicon_size(qicon: QIcon) -> int:
    """估算QIcon的内存占用大小"""
    try:
        # 获取不同尺寸的pixmap并估算
        sizes = [16, 32, 64, 96, 128, 256]
        total_size = 0
        for size in sizes:
            pm = qicon.pixmap(size, size)
            if not pm.isNull():
                total_size += estimate_qpixmap_size(pm)
        return total_size + asizeof.asizeof(qicon)
    except Exception:
        pass
    return asizeof.asizeof(qicon)
