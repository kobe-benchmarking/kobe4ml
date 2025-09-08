import os
import logging
import json
import pickle
import numpy as np

def get_logger(level='DEBUG'):
    """
    Create and configure a logger object with the specified logging level.

    :param level: Logging level to set for the logger. Default is 'DEBUG'.
    :return: Logger object configured with the specified logging level.
    """
    logger = logging.getLogger(__name__)

    level_name = logging.getLevelName(level)
    logger.setLevel(level_name)
    
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

def save_pickle(obj, path):
    """
    Save a Python object to a pickle file.

    :param obj: Python object to save.
    :param path: File path where to save the pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """
    Load a Python object from a pickle file.

    :param path: File path of the pickle.
    :return: Python object loaded from pickle.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_np(data, path):
    """
    Save a NumPy array to a .npy file.

    :param data: NumPy array to save.
    :param path: Full path to save the array (should end with .npy).
    """
    np.save(path, data)

def load_np(path):
    """
    Load a NumPy array from a .npy file.

    :param path: Full path to the .npy file.
    :return: Loaded NumPy array.
    """
    return np.load(path)