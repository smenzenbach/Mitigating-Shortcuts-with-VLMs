import logging
import sys


class MyLogger:
    def __init__(self, logging_file):
        self.logger = logging.getLogger(logging_file)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CustomFormatter())
        self.logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(logging_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(CustomFormatter())
        self.logger.addHandler(file_handler)


class CustomFormatter(logging.Formatter):

    green = "\x1b[32;20m"
    blue = "\x1b[1;34;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s  %(levelname)s  %(message)s"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
