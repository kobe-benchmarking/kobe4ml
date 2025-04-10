import logging

def get_logger(level='DEBUG'):
    """
    Create and configure a logger object with the specified logging level.

    :param level: Logging level to set for the logger. Default is 'DEBUG'.
    :return: Logger object configured with the specified logging level.
    """
    logger = logging.getLogger(__name__)

    level_name = logging.getLevelName(level)
    logger.setLevel(level_name)
    
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger