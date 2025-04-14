import torch
import multiprocessing
import pandas as pd
import numpy as np
import dask.dataframe as dd
from torch.utils.data import Dataset, DataLoader
import os
import io
import s3fs
import json
import mne

from . import utils

logger = utils.get_logger(level='INFO')

def get_boas_data(base_path, output_path):
    fs = s3fs.S3FileSystem(anon=False)

    for subject_folder in fs.glob(f'{base_path}/sub-*'):
        subject_id = os.path.basename(subject_folder)
        eeg_folder = f'{subject_folder}/eeg'

        output_file = f"{output_path}/{subject_id}.csv"

        if fs.exists(output_file):
            continue

        if not fs.exists(eeg_folder):
            logger.debug(f'No EEG folder found for {subject_id}. Skipping.')
            continue

        eeg_file_pattern = f'{eeg_folder}/{subject_id}_task-Sleep_acq-headband_eeg.edf'
        events_file_pattern = f'{eeg_folder}/{subject_id}_task-Sleep_acq-psg_events.tsv'

        try:
            with fs.open(eeg_file_pattern, 'rb') as eeg_file:
                eeg_bytes = io.BytesIO(eeg_file.read()) 
                raw = mne.io.read_raw_edf(eeg_bytes, preload=True)
                x_data = raw.to_data_frame()

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

class TSDataset(Dataset):
    def __init__(self, df, seq_len, X, t, y, per_epoch=True):
        """
        Initializes a time series dataset. It creates sequences from the input data by 
        concatenating features and time columns. The target variable is stored separately.

        :param df: Pandas dataframe containing the data.
        :param seq_len: Length of the input sequence (number of time steps).
        :param X: List of feature columns.
        :param t: List of time-related columns.
        :param y: List of target columns.
        :param per_epoch: Whether to create sequences in non-overlapping (True) or overlapping (False) epochs.
        """
        self.seq_len = seq_len
        self.X = pd.concat([df[X], df[t]], axis=1)
        self.y = df[y]
        self.per_epoch = per_epoch

        logger.debug(f'Initializing dataset with: samples={self.num_samples}, samples/seq={seq_len}, seqs={self.num_seqs}, epochs={self.num_epochs} ')

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        :return: Length of the dataset.
        """
        return self.num_seqs

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: Index of the sample.
        :return: Tuple of features and target tensors.
        """
        if self.per_epoch:
            start_idx = idx * self.seq_len
        else:
            start_idx = idx

        end_idx = start_idx + self.seq_len

        X = self.X.iloc[start_idx:end_idx].values
        y = self.y.iloc[start_idx:end_idx].values

        X, y = torch.FloatTensor(X), torch.LongTensor(y)

        return X, y
    
    @property
    def num_samples(self):
        """
        Returns the total number of samples in the dataset.
        
        :return: Total number of samples.
        """
        return self.X.shape[0]
    
    @property
    def num_epochs(self):
        """
        Returns the number of full epochs available based on the dataset size.

        :return: Number of epochs.
        """
        return self.num_samples // 7680

    @property
    def max_seq_id(self):
        """
        Returns the maximum index for a sequence.

        :return: Maximum index for a sequence.
        """
        return self.num_samples - self.seq_len
    
    @property
    def num_seqs(self):
        """
        Returns the number of sequences that can be created from the dataset.

        :return: Number of sequences.
        """
        if self.per_epoch:
            return self.num_samples // self.seq_len
        else:
            return self.max_seq_id + 1

def split_data(dir, train_size=57, val_size=1, test_size=1):
    """
    Split the CSV files into training, validation, and test sets.

    :param dir: S3 directory containing the CSV files (e.g., 's3://manolo-data/datasets/processed').
    :param train_size: Number of files for training.
    :param val_size: Number of files for validation.
    :param test_size: Number of files for testing.
    :return: Tuple of lists containing CSV file paths for train, val, and test sets.
    """
    fs = s3fs.S3FileSystem(anon=False)

    files = fs.glob(f'{dir}/*.csv')
    
    logger.debug(f'Found {len(files)} files in directory: {dir} ready for splitting.')

    train_paths = files[:train_size]
    val_paths = files[train_size:train_size + val_size]
    test_paths = files[train_size + val_size:train_size + val_size + test_size]

    logger.debug(f'Splitting complete!')

    return (train_paths, val_paths, test_paths)

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

def combine_data(paths, name, seq_len=240, output_s3_path='s3://manolo-data/datasets/bitbrain-ds/proc'):
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

def get_dataframes(paths, seq_len=240, exist=False, output_s3_path='s3://manolo-data/datasets/bitbrain-ds/proc'):
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
    names = ['train', 'val', 'test']
    weights = None

    logger.debug('Creating dataframes for training, validation, and testing.')

    for paths, name in zip(paths, names):
        proc_path = f"{output_s3_path}/{name}.csv"

        if exist and fs.exists(proc_path):
            file_size = fs.info(proc_path)['size']
            logger.debug(f'File size: {file_size} bytes')

            df = dd.read_csv(f's3://{proc_path}', storage_options={'anon': False})
            df = df.compute()

            logger.debug(f'Loaded existing dataframe from {proc_path}.')
        else:
            df = combine_data(paths, name, seq_len)

            if name == 'train':
                logger.debug('Calculating class weights from the training dataframe.')
                weights, _ = extract_weights(df, label_col='majority')

            label_mapping = get_label_mapping(weights=weights)
            df['majority'] = df['majority'].map(label_mapping)

            with fs.open(proc_path, 'w') as file:
                df.to_csv(file, index=False)

            logger.debug(f'Saved {name} dataframe to {proc_path}.')

        dataframes.append(df)

    logger.debug('Dataframes for training, validation, and testing are ready!')

    return tuple(dataframes)

def extract_weights(df, label_col, output_s3_path='s3://manolo-data/datasets/bitbrain-ds/weights.json'):
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

def create_datasets(dataframes, seq_len=7680):
    """
    Create datasets for the specified dataframes (e.g. training, validation, and testing).

    :param dataframes: Tuple of dataframes.
    :param seq_len: Sequence length for each dataset sample.
    :return: Tuple of datasets.
    """
    datasets = []

    X = ['HB_1', 'HB_2']
    t = ['time_norm', 'time', 'seq_id', 'night']
    y = ['majority']

    logger.debug('Creating datasets from dataframes.') 

    for df in dataframes:
        dataset = TSDataset(df, seq_len, X, t, y)
        datasets.append(dataset)

    logger.debug(f'Datasets created successfully!')

    return tuple(datasets)

def create_dataloaders(datasets, batch_size=1, shuffle=[True, False, False], num_workers=None, drop_last=False):
    """
    Create DataLoader objects for the specified datasets, providing data in batches for training, validation, and testing.

    :param datasets: Tuple of datasets.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: List indicating whether to shuffle data for each dataset.
    :param num_workers: Number of subprocesses to use for data loading (default is all available CPU cores).
    :param drop_last: Whether to drop the last incomplete batch.
    :return: Tuple of DataLoader objects.
    """
    dataloaders = []
    cpu_cores = multiprocessing.cpu_count()

    if num_workers is None:
        num_workers = cpu_cores

    logger.debug(f'System has {cpu_cores} CPU cores. Using {num_workers}/{cpu_cores} workers for data loading.')
    
    for dataset, shuffle in zip(datasets, shuffle):
        full_batches = dataset.num_seqs // batch_size

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
        )
        dataloaders.append(dataloader)

        logger.debug(f'Total batches={len(dataloader)} & full batches={full_batches}, with each full batch containing {batch_size} sequences.')
    
    logger.debug('DataLoaders created successfully.')

    return tuple(dataloaders)

def main(url, process, batch_size):
    """
    Main function to preprocess the data.

    :param url: URL of the dataset.
    :param process: Type of process (train/test).
    :param batch_size: Size of the batch for processing.
    :return: Processed dataset.
    """
    logger.info(f"Preprocessing data from URL: {url} with batch size: {batch_size}.")

    samples, chunks = 7680, 32
    seq_len = samples // chunks

    bitbrain_dir = os.path.join(url, 'bitbrain')
    raw_dir = os.path.join(url, 'raw')

    get_boas_data(base_path=bitbrain_dir, output_path=raw_dir)
    datapaths = split_data(dir=raw_dir, train_size=3, val_size=2, test_size=2)

    if process == 'test':
        _, _, test_df = get_dataframes(datapaths, seq_len=seq_len, exist=True)
        datasets = create_datasets(dataframes=(test_df,), seq_len=seq_len)
    elif process == 'train':
        train_df, val_df, _ = get_dataframes(datapaths, seq_len=seq_len, exist=True)
        datasets = create_datasets(dataframes=(train_df, val_df), seq_len=seq_len)
    else:
        raise ValueError(f"Process type '{process}' not recognized")

    dataloaders = create_dataloaders(datasets, batch_size=batch_size, drop_last=False)

    logger.info("Data preprocessing completed successfully")

    return dataloaders

if __name__ == "__main__":
    main()