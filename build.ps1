# 激活虚拟环境
.venv/Scripts/Activate.ps1

nuitka --msvc=latest --standalone --enable-plugin=pyside6 --output-dir=nuitkaFolder --output-filename=picbed.exe --windows-console-mode=attach --python-flag=no_docstrings --windows-icon-from-ico=picbed/image/picbed.ico main.py 
