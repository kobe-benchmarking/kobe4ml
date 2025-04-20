import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
import io
import s3fs
import json
import mne

from . import utils

logger = utils.get_logger(level='DEBUG')

def get_boas_data(base_path, output_path):
    fs = s3fs.S3FileSystem(anon=False)

    for subject_folder in fs.glob(f'{base_path}/sub-*'):
        subject_id = os.path.basename(subject_folder)
        logger.debug(f'Processing subject folder: {subject_folder} with ID: {subject_id}')

        eeg_folder = f'{subject_folder}/eeg'

        output_file = f"{output_path}/{subject_id}.csv"
        logger.debug(f'Output file path: {output_file}')

        if fs.exists(output_file):
            logger.debug(f'Output file {output_file} already exists. Skipping.')
            continue

        if not fs.exists(eeg_folder):
            logger.debug(f'No EEG folder found for {subject_id}. Skipping.')
            continue

        eeg_file_pattern = f'{eeg_folder}/{subject_id}_task-Sleep_acq-headband_eeg.edf'
        events_file_pattern = f'{eeg_folder}/{subject_id}_task-Sleep_acq-psg_events.tsv'

        try:
            with fs.open(eeg_file_pattern, 'rb') as eeg_file:

                eeg_content = eeg_file.read()
                logger.debug(f'EEG file size: {len(eeg_content)} bytes')

                with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                    tmp_file.write(eeg_content)
                    logger.debug(f'Temporary file created at: {tmp_file.name}')
                    logger.debug(f"Temporary file is an eeg file: {tmp_file.name.endswith('.edf')}")

                    tmp_file_path = tmp_file.name

                raw = mne.io.read_raw_edf(tmp_file_path, preload=True)
                logger.debug(f'Raw EEG data is not empty: {raw is not None}')

                x_data = raw.to_data_frame()
                logger.debug(f'EEG data shape for {subject_id}: {x_data.shape}')

            logger.debug(f'x_data shape for {subject_id}: {x_data.shape}')
            logger.debug(f'x_data sample:\n{x_data.head()}')

        except Exception as e:
            logger.debug(f'Error loading EEG data for {subject_id}: {e}')
            continue

        try:
            with fs.open(events_file_pattern, 'r') as events_file:
                y_data = pd.read_csv(events_file, delimiter='\t')

            logger.debug(f'y_data shape for {subject_id}: {y_data.shape}')
            logger.debug(f'y_data sample:\n{y_data.head()}')

        except Exception as e:
            logger.debug(f'Error loading events data for {subject_id}: {e}')
            continue

        y_expanded = pd.DataFrame(index=x_data.index, columns=y_data.columns)

        for _, row in y_data.iterrows():
            begsample = row['begsample'] - 1
            endsample = row['endsample'] - 1
            y_expanded.loc[begsample:endsample] = row.values

        combined_data = pd.concat([x_data, y_expanded], axis=1)

        with fs.open(output_file, 'w') as output_s3_file:
            combined_data.to_csv(output_s3_file, index=False)

        logger.debug(f'Saved combined data for {subject_id} to {output_file}')

def split_data(dir, train_size, test_size):
    """
    Split the CSV files into training and test sets.

    :param dir: S3 directory containing the CSV files (e.g., 's3://manolo-data/datasets/processed').
    :param train_size: Number of files for training.
    :param test_size: Number of files for testing.
    :return: Tuple of lists containing CSV file paths for train and test sets.
    """
    fs = s3fs.S3FileSystem(anon=False)

    files = fs.glob(f'{dir}/*.csv')
    
    logger.debug(f'Found {len(files)} files in directory: {dir} ready for splitting.')

    train_paths = [f's3://{file}' for file in files[:train_size]]
    test_paths = [f's3://{file}' for file in files[train_size:train_size + test_size]]

    logger.debug(f'Splitting complete!')
    logger.debug(f'Train paths: {train_paths}')
    logger.debug(f'Test paths: {test_paths}')

    return (train_paths, test_paths)

def load_file(path):
    """
    Load data from a CSV file (local or S3).

    :param path: Path to the CSV file (can be an S3 path).
    :return: Tuple (X, t, y) where X contains EEG features, t contains time, and y contains labels.
    """
    fs = s3fs.S3FileSystem(anon=False)

    if path.startswith('s3://'):
        with fs.open(path, 'r') as file:
            df = pd.read_csv(file)
    else:
        df = pd.read_csv(path)

    X = df[['HB_1', 'HB_2']].values
    t = df['time'].values
    y = df['majority'].values

    return X, t, y

def combine_data(paths, name, seq_len, output_s3_path):
    """
    Combine data from multiple CSV files into a dataframe, processing sequences and removing invalid rows.

    :param paths: List of file paths to CSV files (can be S3 paths).
    :param name: Context of the data, such as 'train', 'val' or 'test'.
    :param seq_len: Sequence length for grouping data.
    :param output_s3_path: S3 directory where processed data should be stored.
    :return: Combined dataframe after processing.
    """
    fs = s3fs.S3FileSystem(anon=False)

    dataframes, dataframes_8 = [], []
    total_removed_majority = 0

    logger.debug(f'Combining data from {len(paths)} files.')

    for path in paths:
        X, t, y = load_file(path)

        df = pd.DataFrame(X, columns=['HB_1', 'HB_2'])
        df['majority'] = y
        df['time'] = t

        df['seq_id'] = (np.arange(len(df)) // seq_len) + 1
        df['night'] = int(os.path.basename(path).split('-')[1].split('.')[0])
        df['time_norm'] = t

        rows_8 = df[df['majority'] == 8]

        if not rows_8.empty and name=='test':
            dataframes_8.append(rows_8)

        rows_before_majority_drop = df.shape[0]
        df.drop(df[df['majority'] == 8].index, inplace=True)
        total_removed_majority += (rows_before_majority_drop - df.shape[0])

        dataframes.append(df)
    
    logger.debug(f'Removed {total_removed_majority} rows with majority value 8.')

    if name=='test':
        if dataframes_8:
            df_8 = pd.concat(dataframes_8, ignore_index=True)
            proc_path_8 = f"{output_s3_path}/{name}8.csv"

            with fs.open(proc_path_8, 'w') as file:
                df_8.to_csv(file, index=False)

            logger.debug(f'Saved {df_8.shape[0]} rows with majority=8 to {proc_path_8}.')
        else:
            logger.debug(f"The dataframe contains no rows with majority=8, therefore no {name}8.csv was saved.")

    df = pd.concat(dataframes, ignore_index=True)
    logger.debug(f'Combined dataframe shape: {df.shape}')
    
    rows_before_nan_drop = df.shape[0]
    df.dropna(inplace=True)
    logger.debug(f'Removed {rows_before_nan_drop - df.shape[0]} rows with NaN values.')

    assert not df.isna().any().any(), 'NaN values found in the dataframe!'

    stats_path = f"{output_s3_path}/stats.json"
    df = utils.robust_normalize(df, exclude=['night', 'seq_id', 'time', 'majority'], path=stats_path)

    return df

def get_dataframes(paths, seq_len, exist, output_s3_path):
    """
    Create or load dataframes for training, validation, and testing.

    :param paths: List of file paths for training, validation, and testing.
    :param seq_len: Sequence length for processing.
    :param exist: Boolean flag indicating if the dataframes already exist.
    :param output_s3_path: S3 path where processed CSV files should be stored.
    :return: Tuple of dataframes for train, validation, and test sets.
    """
    fs = s3fs.S3FileSystem(anon=False)
    dataframes = []
    names = ['train', 'test']
    weights = None
    weights_path = f"{output_s3_path}/weights.json"

    logger.debug('Creating dataframes for training and testing.')

    for paths, name in zip(paths, names):
        proc_path = f"{output_s3_path}/{name}.csv"

        if exist and fs.exists(proc_path):
            file_size = fs.info(proc_path)['size']
            logger.debug(f'File size: {file_size} bytes')

            df = dd.read_csv(f's3://{proc_path}', storage_options={'anon': False})
            df = df.compute()

            logger.debug(f'Loaded existing dataframe from {proc_path}.')
        else:
            df = combine_data(paths, name, seq_len, output_s3_path)

            if name == 'train':
                logger.debug('Calculating class weights from the training dataframe.')
                weights, _ = extract_weights(df, label_col='majority', output_s3_path=weights_path)

            label_mapping = get_label_mapping(weights=weights)
            df['majority'] = df['majority'].map(label_mapping)

            with fs.open(proc_path, 'w') as file:
                df.to_csv(file, index=False)

            logger.debug(f'Saved {name} dataframe to {proc_path}.')

        dataframes.append(df)

    logger.debug('Dataframes for training and testing are ready!')

    return tuple(dataframes)

def extract_weights(df, label_col, output_s3_path):
    """
    Calculate class weights from the training dataframe to handle class imbalance, and save them to S3.

    :param df: Dataframe containing the training data.
    :param label_col: The name of the column containing class labels.
    :param output_s3_path: S3 path where weights.json should be stored.
    :return: A tuple containing a dictionary of class weights and a list of class labels if mapping is enabled.
    """
    fs = s3fs.S3FileSystem(anon=False)

    occs = df[label_col].value_counts().to_dict()
    inverse_occs = {key: 1e-10 for key in occs.keys()}

    for key, value in occs.items():
        inverse_occs[int(key)] = 1 / (value + 1e-10)

    weights = {key: value / sum(inverse_occs.values()) for key, value in inverse_occs.items()}
    weights = dict(sorted(weights.items()))

    new_weights = {i: weights[key] for i, key in enumerate(weights.keys())}

    weights_json = json.dumps(new_weights, indent=4)
    with fs.open(output_s3_path, 'w') as f:
        f.write(weights_json)
    
    logger.debug(f'Saved class weights to {output_s3_path}.')

    return weights, new_weights

def get_label_mapping(weights):
    label_mapping = {original_label: new_index for new_index, original_label in enumerate(weights.keys())}

    return label_mapping

def create_dataset(dataframe):
    """
    Create dataset for the specified dataframe (e.g. training and testing).

    :param dataframe: Dataframe containing EEG data.
    :return: X, y - numpy arrays with the dataset.
    """
    logger.debug('Creating dataset from dataframe.')

    X_list = []
    y_list = []

    X_columns = ['HB_1', 'HB_2']
    y_column = 'majority'

    for idx in range(len(dataframe)):
        X_data = dataframe[X_columns].iloc[idx].values
        y_data = dataframe[y_column].iloc[idx]

        X_list.append(X_data)
        y_list.append(y_data)

    X = np.array(X_list)
    y = np.array(y_list)

    logger.debug(f'Datasets created successfully! Shapes -> X: {X.shape}, y: {y.shape}')
    
    return X, y

def main(in_url, out_url, process, train_size, test_size, seq_len):
    """
    Main function to preprocess the data.

    :param in_url: Input URL for the dataset.
    :param out_url: Output URL for the processed dataset.
    :param process: Type of process (prepare/work).
    :return: Processed dataset.
    """
    seq_len = 7680*2

    logger.info(f"Preprocessing data from URL: {in_url}.")

    bitbrain_dir = os.path.join(in_url, 'bitbrain')
    raw_dir = os.path.join(out_url, 'raw')
    proc_dir = os.path.join(out_url, 'proc')

    get_boas_data(base_path=bitbrain_dir, output_path=raw_dir)
    datapaths = split_data(dir=raw_dir, train_size=train_size, test_size=test_size)

    if process == 'work':
        _, df = get_dataframes(datapaths, 
                               seq_len=seq_len, 
                               exist=False, 
                               output_s3_path=proc_dir,)
        
    elif process == 'prepare':
        df, _ = get_dataframes(datapaths, 
                               seq_len=seq_len, 
                               exist=False, 
                               output_s3_path=proc_dir)

    else:
        raise ValueError(f"Process type '{process}' not recognized")
    
    dataset = create_dataset(dataframe=df)

    logger.info("Data preprocessing completed successfully")

    return dataset

if __name__ == "__main__":
    main()