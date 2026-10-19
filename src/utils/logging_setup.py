"""Console and UTF-8 file logging with one owned handler pair per invocation."""

import logging
import sys


def get_logger(name, filename):
    filename.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(filename, encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger
