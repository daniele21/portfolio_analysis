import logging
from typing import Text


class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    green = "\x1b[32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name: Text = __name__,
                 logger_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logger_level)

    # create console handler with a higher log level
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(ColorFormatter())

        logger.addHandler(ch)

    return logger
