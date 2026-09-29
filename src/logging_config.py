"""
Centralized logging: every script calls setup_logging(<name>) once and gets
a timestamped log file under logs/ plus console output.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

DEFAULT_MAX_LOG_FILES = 100


def _cleanup_old_logs(log_dir: Path, max_files: int = DEFAULT_MAX_LOG_FILES) -> None:
    """
    Keep only the newest `max_files` logs. Sorted by modification time, NOT
    filename: a filename sort mixes up order across different script
    prefixes and could delete the current run's own open log file (Windows
    then raises PermissionError). Each delete is best-effort.
    """
    log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime)
    if len(log_files) > max_files:
        for old_file in log_files[: len(log_files) - max_files]:
            try:
                old_file.unlink()
            except OSError as e:
                logging.getLogger().warning("Could not delete old log file %s: %s", old_file, e)


def setup_logging(run_name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized for '%s'. Writing to %s", run_name, log_file)
    _cleanup_old_logs(log_path)
    return logger
