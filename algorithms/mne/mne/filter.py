import warnings
import pandas as pd
import mne
from . import utils

logger = utils.get_logger(level='DEBUG')

warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    """
    Main function to filter data in test.csv using MNE's filter_data function
    and save it as estim_mne.csv.
    """
    data_columns = ['HB_1', 'HB_2']

    csv_path = utils.get_path('..', '..', 'data', 'proc', filename='test_unorm.csv')
    mne_csv_path = utils.get_path('..', '..', 'data', 'proc', filename='estim_mne.csv')

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV from {csv_path}")

    data = df[data_columns].to_numpy().T

    filtered_data = mne.filter.filter_data(data, sfreq=256, l_freq=0.5, h_freq=40)
    filtered_df = pd.DataFrame(filtered_data.T, columns=data_columns)

    for col in data_columns:
        df[f'noise_{col}'] = df[col] - filtered_df[col]

    df.to_csv(mne_csv_path, index=False)
    logger.info(f"Filtered data and noise columns saved to {mne_csv_path}.")

if __name__ == '__main__':
    main()