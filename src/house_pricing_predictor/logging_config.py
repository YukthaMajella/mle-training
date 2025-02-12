import os
import logging
import logging.config

from logging_tree import printout

from dummy_module import dummy_function

# Logging Config
# More on Logging Configuration
# https://docs.python.org/3/library/logging.config.html
# Setting up a config
LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """
    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


if __name__ == "__main__":
    # configuring and assigning in the logger can be done by the below function
    logger = configure_logger(log_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),"custom_config.log"))
    logger.info(f"Logging Test - Start")
    logger.info(f"Logging Test - Test 1 Done")
    logger.warning("Watch out!")
    dummy_function()

    # printing out the current loging confiurations being used
    printout()


import logging
import logging.config
import os

def setup_logging(log_level="INFO", log_path=None, console_log=True):
    """
    Set up logging configuration.

    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    :param log_path: Path to log file, or None to log only to console.
    :param console_log: Whether to log to console. Default is True.
    """
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Base log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Handlers list to add
    handlers = []

    if console_log:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
    
    if log_path:
        # File handler if log path is provided
        os.makedirs(os.path.dirname(log_path), exist_ok=True)  # Ensure directory exists
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Set up the root logger with all handlers
    logging.basicConfig(level=log_level, handlers=handlers)
    
    # Log a message that logging has been set up
    logging.info("Logging setup completed.")
