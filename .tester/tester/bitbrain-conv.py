import os
import pandas as pd
import glob
import mne

import converter

logger = converter.get_logger(level='INFO')

def bitbrain(dir):
    """
    Load the Bitbrain dataset from the specified directory, process it and return a dataframe along with metadata.

    :param dir: Directory containing the Bitbrain dataset.
    :return: A tuple containing the processed data as numpy arrays and a list of metadata features.
    """
    all_data = []

    columns = ['HB_1', 'HB_2', 'time', 'majority', 'night', 'onset', 'duration', 'begsample', 'endsample', 'offset', 'ai_psg', 'HB_IMU_1', 'HB_IMU_2', 'HB_IMU_3', 'HB_IMU_4', 'HB_IMU_5', 'HB_IMU_6', 'HB_PULSE']
    features = ['HB_1', 'HB_2']
    time = ['time']
    labels = ['majority']
    split = ['night']
    weights = ['majority']
    sort = ['night', 'time']
    other = ['night', 'onset', 'duration', 'begsample', 'endsample', 'offset', 'ai_psg', 'HB_IMU_1', 'HB_IMU_2', 'HB_IMU_3', 'HB_IMU_4', 'HB_IMU_5', 'HB_IMU_6', 'HB_PULSE']

    for subject_folder in glob.glob(os.path.join(dir, 'sub-*')):
        subject_id = os.path.basename(subject_folder)
        eeg_folder = os.path.join(subject_folder, 'eeg')

        if not os.path.exists(eeg_folder):
            continue

        eeg_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-headband_eeg.edf')
        events_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-psg_events.tsv')

        try:
            raw = mne.io.read_raw_edf(eeg_file_pattern, preload=True)
            x_data = raw.to_data_frame()
        except Exception as e:
            continue

        try:
            y_data = pd.read_csv(events_file_pattern, delimiter='\t')
        except Exception as e:
            continue

        y_expanded = pd.DataFrame(index=x_data.index, columns=y_data.columns)

        for _, row in y_data.iterrows():
            begsample = row['begsample'] - 1
            endsample = row['endsample'] - 1

            y_expanded.loc[begsample:endsample] = row.values

        combined_data = pd.concat([x_data, y_expanded], axis=1)
        combined_data['night'] = int(subject_id.replace('sub-', ''))

        all_data.append(combined_data)

    df = pd.concat(all_data, ignore_index=True)

    before_nan_drop = len(df)
    df = df.dropna(subset=features + time + labels)
    dropped_na = before_nan_drop - len(df)

    before_majority_drop = len(df)
    df = df[df['majority'] != 8]
    dropped_majority_8 = before_majority_drop - len(df)

    logger.info(f"Dropped {dropped_na} rows containing NaN values.")
    logger.info(f"Dropped {dropped_majority_8} rows where majority == 8.")
    
    logger.info("Preview of combined dataframe:")
    logger.info(df.head().to_string())

    unique_nights = df['night'].unique()
    logger.info(f"Unique nights in the combined dataframe: {sorted(unique_nights)}")

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
    Main function to load the Bitbrain dataset, convert it to numpy format, and save metadata as JSON.
    """
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))

    in_dir = converter.get_dir(root, 'DATASETS', 'bitbrain_small')
    out_dir = converter.get_dir(root, 'DATASETS', 'bitbrain_conv')

    out_npz_path = converter.get_path(out_dir, filename='bitbrain.npz')
    out_json_path = converter.get_path(out_dir, filename='bitbrain.json')
    
    logger.info(f"Loading data from bitbrain dataset and returning it as a dataframe along with metadata.")
    data, keys = bitbrain(dir=in_dir)

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