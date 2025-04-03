import json
import s3fs
    
def robust_normalize(df, exclude, path):
    """
    Normalize data using robust scaling (median and IQR) from precomputed stats.

    :param df: DataFrame containing the data to normalize.
    :param exclude: List of columns to exclude from normalization.
    :param path: File path to save the computed statistics.
    :return: Processed DataFrame with normalized data.
    """
    fs = s3fs.S3FileSystem(anon=False)
    newdf = df.copy()

    stats = get_stats(df)

    stats_json = json.dumps(stats, indent=4)
    with fs.open(path, 'w') as f:
        f.write(stats_json)
    
    for col in df.columns:
        if col not in exclude:
            median = stats[col]['median']
            iqr = stats[col]['iqr']
            
            newdf[col] = (df[col] - median) / (iqr if iqr > 0 else 1)

    return newdf

def get_stats(df):
    """
    Compute mean, standard deviation, median, and IQR for each column in the DataFrame.

    :param df: DataFrame containing the data to compute statistics for.
    :return: Dictionary containing statistics for each column.
    """
    stats = {}

    for col in df.columns:
        series = df[col]

        mean = series.mean()
        std = series.std()
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)

        stats[col] = {
            'mean': mean,
            'std': std,
            'median': median,
            'iqr': iqr
        }

    return stats