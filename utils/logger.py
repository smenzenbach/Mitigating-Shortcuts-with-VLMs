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
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
        )
        self.logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(logging_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
        )
        self.logger.addHandler(file_handler)
