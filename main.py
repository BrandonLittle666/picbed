import sys
from pathlib import Path

from loguru import logger

try:
    from picbed.singleton import cleanup_singleton, initSingleton
    shm, has_running_instance = initSingleton()
    if has_running_instance:
        sys.exit(0)
    
    from picbed.utils import setup_stdout
    setup_stdout()

    import picbed.app
    picbed.app.main()

    cleanup_singleton(shm)

    logger.info("PicBed Exited.")

except Exception as e:
    try:
        from picbed.utils import appdir
        logdir = appdir() / "logs"
    except Exception:
        logdir = Path("logs")
    logger.add(logdir / "picbed-critical.log", rotation="10 MB", retention=10, enqueue=True)
    logger.exception(e)
    logger.info(f"PicBed Exited with exception. {e}")
    sys.exit(0)
