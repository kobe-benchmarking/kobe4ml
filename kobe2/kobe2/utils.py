import yaml
import logging
import os

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

def get_dir(*sub_dirs):
    """
    Retrieve or create a directory path based on the script's location and the specified subdirectories.

    :param sub_dirs: List of subdirectories to append to the script's directory.
    :return: Full path to the directory.
    """
    dir = os.path.join(*sub_dirs)

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def get_path(*dirs, filename):
    """
    Construct a full file path by combining directory paths and a filename.

    :param dirs: List of directory paths.
    :param filename: Name of the file.
    :return: Full path to the file.
    """
    dir_path = get_dir(*dirs)
    path = os.path.join(dir_path, filename)

    return path    

def load_yaml(path):
    """
    Load a YAML file and return its contents as a dictionary.
    
    :param path: Path to the YAML file.
    :return: Dictionary containing the contents of the YAML file.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def split_path(path):
    """
    Split a path into (directories, filename), based on the last '/'.
    """
    dirs, filename = os.path.split(path)
    
    return dirs, filename