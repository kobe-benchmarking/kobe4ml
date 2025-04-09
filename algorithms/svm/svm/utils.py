import os
import boto3
from io import BytesIO
import pickle

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

def save_model_to_s3(model, model_url):
    """
    Save the trained model to an S3 bucket.
    
    :param model: The model to be saved (SVM).
    :param model_url: URL reference for the S3 location (e.g., 's3://bucket-name/path/to/model.pkl').
    """
    s3 = boto3.client('s3')
    s3_parts = model_url.replace('s3://', '').split('/', 1)
    bucket_name = s3_parts[0]
    key = s3_parts[1]

    model_byte_stream = BytesIO()
    pickle.dump(model, model_byte_stream)
    model_byte_stream.seek(0)

    s3.upload_fileobj(model_byte_stream, bucket_name, key)
    print(f"Model saved to {model_url}")
    
def load_model_from_s3(url):
    """
    Load the SVM model directly from S3 into memory.
    
    :param url: S3 URL of the model (e.g., 's3://bucket-name/path/to/model.pkl')
    :return: Loaded SVM model
    """
    s3 = boto3.client('s3')
    s3_parts = url.replace('s3://', '').split('/', 1)
    bucket_name = s3_parts[0]
    key = s3_parts[1]

    model_byte_stream = BytesIO()
    s3.download_fileobj(bucket_name, key, model_byte_stream)
    model_byte_stream.seek(0)

    model = pickle.load(model_byte_stream)

    return model