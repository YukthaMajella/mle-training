import logging
import logging.config
import os

def setup_logging(log_level="INFO", log_path=None, console_log=True):

    log_level = getattr(logging, log_level.upper(), logging.INFO)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = []

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, handlers=handlers)

    logging.info("Logging setup completed.")
