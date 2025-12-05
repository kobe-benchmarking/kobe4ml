import os
import pandas as pd
import glob
import mne

import converter

logger = converter.get_logger(level='INFO')

def jetengines(dir):
    """
    Load the jet_engines dataset from the specified directory, process it and return a dataframe along with metadata.

    :param dir: Directory containing the jet_engines dataset.
    :return: A tuple containing the processed data as numpy arrays and a list of metadata features.
    """
    columns = ['TSid','Timestamp','DNH','DWf','DPt13','DTt13','DPt31','DTt31','DTt5','DPt45','label','FM','Ratio','DSW','DSE','Alt','dTisa','M','Mode','CntrDmd','label-2']
    features = ['DNH','DWf','DPt13','DTt13','DPt31','DTt31','DTt5','DPt45']
    time = ['Timestamp']
    labels = ['label-2']
    split = ['TSid']
    weights = ['label-2']
    sort = ['TSid', 'Timestamp']
    other = ['label','FM','Ratio','DSW','DSE','Alt','dTisa','M','Mode','CntrDmd']

    # Read all CSV files from the directory
    csv_files = glob.glob(os.path.join(dir, '*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {dir}")
    
    logger.info(f"Found {len(csv_files)} CSV file(s) in {dir}")
    
    all_dfs = []
    expected_columns = None
    
    for csv_file in csv_files:
        try:
            temp_df = pd.read_csv(csv_file)
            logger.info(f"Loaded {csv_file} with {len(temp_df)} rows")
            
            # Check if columns match exactly (same names and same order)
            if expected_columns is None:
                expected_columns = list(temp_df.columns)
            elif list(temp_df.columns) != expected_columns:
                raise ValueError(f"Column mismatch in {csv_file}. Expected columns: {expected_columns}, Got: {list(temp_df.columns)}")
            
            all_dfs.append(temp_df)
        except Exception as e:
            logger.error(f"Failed to load CSV file {csv_file}: {e}")
            raise
    
    # Concatenate all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Successfully concatenated {len(csv_files)} CSV file(s) with total {len(df)} rows")
    
    # Sort by TSid and Timestamp
    df = df.sort_values(by=['TSid', 'Timestamp']).reset_index(drop=True)
    logger.info(f"Sorted dataframe by TSid and Timestamp")

    before_nan_drop = len(df)
    df = df.dropna(subset=features + time + labels)
    dropped_na = before_nan_drop - len(df)

    # before_majority_drop = len(df)
    # df = df[df['majority'] != 8]
    # dropped_majority_8 = before_majority_drop - len(df)

    logger.info(f"Dropped {dropped_na} rows containing NaN values.")
    # logger.info(f"Dropped {dropped_majority_8} rows where majority == 8.")
    
    logger.info("Preview of combined dataframe:")
    logger.info(df.head().to_string())

    unique_time_series = df['TSid'].unique()
    logger.info(f"Unique time series in the combined dataframe: {sorted(unique_time_series)}")

    features_npy = df[features].values.astype('float32')
    time_npy = df[time].values.astype('float32')
    labels_npy = df[labels].values.astype('int32')
    split_npy = df[split].values.astype('int32')
    weights_npy = df[weights].values.astype('float32')
    sort_npy = df[sort].values.astype('float32')
    other_npy = df[other].values.astype('float32')

    return (features_npy, time_npy, labels_npy, split_npy, weights_npy, sort_npy, other_npy), (columns, features, time, labels, split, weights, sort, other)

def main():
    """
    Main function to load the jet_engines dataset, convert it to numpy format, and save metadata as JSON.
    """
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

    in_dir = converter.get_dir(root, 'DATASETS', 'jet_engines_init')
    out_dir = converter.get_dir(root, 'DATASETS', 'jet_engines_conv')

    out_npz_path = converter.get_path(out_dir, filename='jet_engines.npz')
    out_json_path = converter.get_path(out_dir, filename='jet_engines.json')
    
    logger.info(f"Loading data from jet_engines dataset and returning it as a dataframe along with metadata.")
    data, keys = jetengines(dir=in_dir)

    logger.info(f"Converting dataframe to numpy array.")
    converter.create_npz(data=data, 
                         path=out_npz_path)

    logger.info(f"Creating metadata JSON file.")
    converter.create_metadata(
        path=out_json_path,
        keys=keys)

    logger.info(f"Conversion finished!")

if __name__ == "__main__":
    main()