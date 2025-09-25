from __future__ import annotations

import json
import mimetypes
import os
import socket
import threading
import urllib.parse as parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from loguru import logger
from PySide6.QtCore import QObject, Signal

from picbed.s3_client import S3Client
from picbed.utils import auto_rename, format_size, render_placeholders


class UploadHandler(BaseHTTPRequestHandler):
    s3_client: S3Client | None = None
    server: PicGoLikeServer | None = None

    def _set_json_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def log_request(self, code='-', size='-'):
        """ Ignore request log """
        pass

    def do_POST(self) -> None:
        '''
        All post requests to server will be handled by this method.
        '''
        # record request source and basic information
        client_ip = self.client_address[0] if self.client_address else "unknown"
        user_agent = self.headers.get('User-Agent', 'unknown')
        content_type = self.headers.get('Content-Type', 'unknown')
        length = int(self.headers.get('Content-Length', 0) or 0)
        
        logger.debug(f"Got POST request - Client IP: {client_ip}, User-Agent: {user_agent}, Content-Type: {content_type}, Content-Length: {length}, Path: {self.path}")

        # region check before handling
        # check if the request is too large
        if length > UploadHandler.s3_client._config.max_upload_size:
            msg = f"Request too large. Max upload size is {format_size(UploadHandler.s3_client._config.max_upload_size)}, got {format_size(length)}."
            self._set_json_headers(413)
            self.wfile.write(json.dumps({
                "success": False,
                "result": [],
                "message": msg
            }).encode("utf-8"))
            logger.warning(msg)
            return

        # parse query params
        path_only = self.path.split("?")[0]
        if path_only.rstrip("/") != "/upload":
            self._set_json_headers(404)
            self.wfile.write(json.dumps({
                "success": False,
                "result": [],
                "message": "Path Not Supported. Please upload via `/upload` path."
            }).encode("utf-8"))
            return
        
        rawdata = self.rfile.read(length) if length > 0 else b''

        if not rawdata:
            self._set_json_headers(400)
            self.wfile.write(json.dumps({
                "success": False,
                "result": [],
                "message": "No data provided."
            }).encode("utf-8"))
            return
        # endregion

        # region handle upload bytes
        if content_type == "application/octet-stream":
            suggested_name : str | None = None
            if '?' in self.path:
                query_params = parse.parse_qs(self.path.split("?")[1])
                filename_params = query_params.get("filename") or query_params.get("FileName") or query_params.get("Filename")
                if filename_params:
                    suggested_name = filename_params[0]
            try:
                result = self._handle_upload_bytes(rawdata, suggested_name=suggested_name)
                self._set_json_headers(200)
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                logger.exception(f"Upload processing exception: {e} (Source: {client_ip})")
                self._set_json_headers(500)
                self.wfile.write(json.dumps({
                    "success": False,
                    "result": [],
                    "message": str(e)
                }).encode("utf-8"))
            return
        # endregion
        
        # region handle upload paths
        try:
            # expected format: {list: ['xxx.jpg']}
            data: dict[str, Any] = json.loads(rawdata)
            paths: list[str] = data["list"]
            logger.info(f"解析请求参数成功 - 文件路径数量: {len(paths)}, 路径列表: {paths} (来源: {client_ip})")
        except Exception as e:
            logger.error(f"解析JSON数据失败: {e} (来源: {client_ip}, 数据: {rawdata.decode('utf-8', errors='ignore')})")
            self._set_json_headers(400)
            self.wfile.write(json.dumps({
                "success": False,
                "result": [],
                "message": "Invalid JSON data. Expected format: {list: ['xxx.jpg']}"
            }).encode("utf-8"))
            return

        if not paths:
            logger.warning(f"请求参数为空 - 没有提供文件路径 (来源: {client_ip})")
            self._set_json_headers(400)
            self.wfile.write(json.dumps({
                "success": False,
                "result": [],
                "message": "No paths provided. Expected format: {list: ['xxx.jpg']}"
            }).encode("utf-8"))
            return

        try:
            logger.info(f"开始处理上传请求 - 文件数量: {len(paths)} (来源: {client_ip})")
            result = self._handle_upload_paths(paths)
            
            if result["success"]:
                logger.info(f"上传成功 - 成功上传: {len(result['result'])} 个文件, 结果: {result['result']} (来源: {client_ip})")
            else:
                logger.warning(f"上传失败 - 错误信息: {result['message']} (来源: {client_ip})")
            
            self._set_json_headers(200)
            self.wfile.write(json.dumps(result).encode("utf-8"))
            if UploadHandler.server is not None:
                UploadHandler.server.path_uploaded.emit(result)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"上传处理异常: {e} (来源: {client_ip})")
            self._set_json_headers(500)
            self.wfile.write(json.dumps({
                "success": False,
                "result": [],
                "message": str(e)
            }).encode("utf-8"))
        # endregion

    def _handle_upload_paths(self, paths: list[str]) -> dict:
        '''
        Upload paths to S3.

        Args:
            paths: list of paths to upload
        Returns:
            dict with success, result, and message
        '''

        if UploadHandler.s3_client is None:
            raise RuntimeError("S3 client is not configured")
        
        result: list[str] = []
        errors: list[str] = []
        for path in paths:
            try:
                mime: str | None = None
                if os.path.isfile(path):
                    with open(path, "rb") as f:
                        data = f.read()
                        mime = mimetypes.guess_type(path)[0]
                elif path.lower().startswith("http"):
                    with urllib.request.urlopen(path, timeout=4) as f:
                        data = f.read()
                        mime = mimetypes.guess_type(path)[0]
                else:
                    errors.append(f"Invalid path: {path}")
                    continue
            except Exception as e:
                errors.append(f"Failed to upload {path}: {e}")
                continue
            
            if mime:
                orig = os.path.basename(path)
                key = UploadHandler.s3_client.upload_bytes(data, orig, content_type=mime, public=True)
                url = UploadHandler.s3_client.public_url_for_output(key, original_name=orig)
                result.append(url)
                continue
            if mime is None:
                orig = os.path.basename(path)
                key = UploadHandler.s3_client.upload_bytes(data, orig, content_type=mime, public=True)
                url = UploadHandler.s3_client.public_url_for_output(key, original_name=orig)
                result.append(url)
                continue
        if not result:
            if not errors:
                return {"success": False, "result": [], "message": "No image or file in clipboard"}
            else:
                return {"success": False, "result": [], "message": "\n".join(errors)}
        else:
           return {"success": True, "result": result, "message": "\n".join(errors)}

    def _handle_upload_bytes(self, data: bytes, suggested_name: str | None = None) -> dict:
        if UploadHandler.s3_client is None:
            raise RuntimeError("S3 client is not configured")
        # generate key
        if self.s3_client.upload_path_template:
            if not suggested_name:
                suggested_name = "upload.png"
            key = render_placeholders(
                self.s3_client.upload_path_template,
                original_name=suggested_name,
                content_bytes=data,
            )
        elif not suggested_name:
            key = auto_rename('upload.png', data)
        else:
            key = suggested_name
        mime = mimetypes.guess_type(key)[0]
        key = UploadHandler.s3_client.upload_bytes(data, key, content_type=mime, public=True)
        url = UploadHandler.s3_client.public_url_for_output(key, original_name=key)
        return {"success": True, "result": [url], "message": ""}


class PicGoLikeServer(QObject):
    path_uploaded = Signal(object)
    
    def __init__(self, s3_client: S3Client, host: str = "127.0.0.1", port: int = 36677) -> None:
        super().__init__()
        self._host = host
        self._port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._s3 = s3_client

        # check if host is ipv6
        if ":" in host:
            ThreadingHTTPServer.address_family = socket.AF_INET6
        else:
            ThreadingHTTPServer.address_family = socket.AF_INET

    def start(self) -> None:
        if self._running:
            return
        UploadHandler.s3_client = self._s3
        UploadHandler.server = self
        self._server = ThreadingHTTPServer((self._host, self._port), UploadHandler)
        self._server.daemon_threads = True

        def _serve():
            try:
                logger.info(f"PicBed server listening on http://{self._host}:{self._port}")
                self._server.serve_forever(poll_interval=0.5)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"PicBed server stopped: {e}")

        self._thread = threading.Thread(target=_serve, name="PicBedServer", daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        try:
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
                UploadHandler.server = None
                UploadHandler.s3_client = None
        except Exception:
            pass
        self._running = False
        logger.info("PicBed server stopped")

    def set_s3_client(self, s3_client: S3Client) -> None:
        if self._s3 == s3_client:
            return
        self._s3 = s3_client
        UploadHandler.s3_client = s3_client
        UploadHandler.server = self
