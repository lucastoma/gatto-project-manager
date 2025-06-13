"""
Logger module for LAB Color Transfer algorithm.
"""
import logging


def get_logger(name: str = None) -> logging.Logger:
    """
    Returns a configured logger instance.
    """
    logger_name = name or 'lab_transfer'
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
