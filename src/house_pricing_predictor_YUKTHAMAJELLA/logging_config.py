"""
This module contains the function of the logging process for the House Pricing Predictor
project.

It contains functions to configure the log level, log path and console log.

"""

import logging
import logging.config
import os


def setup_logging(log_level="INFO", log_path=None, console_log=True):
    """
    Configures logging settings for the application.

    Parameters
    ----------
    log_level : str, optional
        The logging level to use. Default is 'INFO'.

    log_path : str, optional
        The path to a log file. If None, no log file will be created. Default is None.

    console_log : bool, optional
        If True, log messages will be displayed in the console. Default is True.

    Returns
    -------
    None
        This function does not return any value. It sets up logging to handle messages
        based on the given configuration.
    """

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
