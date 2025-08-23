import json
import logging
import numpy as np
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(script_dir, *sub_dirs)

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

def load_npy(path):
    """
    Load a .npy file from the given path.

    :param path: Full path to the .npy file.
    :return: Loaded numpy array.
    """
    return np.load(path)

def save_npy(data, path):
    """
    Save a numpy array to a .npy file in the specified directory.

    :param data: Numpy array to save.
    :param path: Full path to the .npy file.
    """
    np.save(path, data)

def load_npz(path):
    """
    Load a structured .npz file from the given path.

    :param path: Full path to the .npz file.
    :return: Loaded numpy structured array.
    """
    return dict(np.load(path))

def save_npz(data, path):
    """
    Save a structured numpy array to a .npz file at the specified path.

    :param data: Structured numpy array to save.
    :param path: Full path to the .npz file.
    """
    np.savez(path, **data)

def load_json(path):
    """
    Load a JSON file from the given path.

    :param path: Full path to the .json file.
    :return: Parsed JSON as a Python dict.
    """
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    """
    Save a Python dict to a JSON file at the specified path.

    :param data: Data to save as JSON.
    :param path: Full path to the .json file.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)