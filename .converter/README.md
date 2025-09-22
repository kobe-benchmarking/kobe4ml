# Manolo Dataset Format Specification

This dataset format defines a structured way to organize multi-modal time series data in `.npz` format, accompanied by a `.json` metadata file.

- `dataset.npz`: Compressed NumPy archive containing arrays grouped by type.
- `dataset.json`: Metadata file describing column roles and structure.


## Metadata Keys (`dataset.json`)

| Key        | Type     | Description                                                                 |
|------------|----------|-----------------------------------------------------------------------------|
| `columns`  | `List[str]` | All distinct columns in the dataset (union of all below). |
| `features` | `List[str]` | Names of input feature columns.                   |
| `time`     | `List[str]` | Names of time-related columns.                                |
| `labels`   | `List[str]` | Names of target columns used for supervised learning.                        |
| `split`    | `List[str]` | Exactly one column used to group data for splitting.        |
| `weights`  | `List[str]` | Subset (or all) of `labels` used for computing class weights during training. |
| `other`    | `List[str]` | All other auxiliary columns not listed in `features`, `time`, or `labels`.   |
| `columns`  | `List[str]` | Concatenation of all above, preserving the full original column order.       |


## Dataset Structure (`dataset.npz`)

Each key corresponds to a NumPy array shaped as follows:

- `features`: (T, Fs) or higher (e.g. (T, C, H, W) for image-like inputs)
- `time`: (T, Ts)
- `labels`: (T, Ls)
- `split`: (T, 1)
- `weights`: (T, Ws)
- `other`: (T, Os)

Here, T = number of total samples.


## Rules & Assumptions

- `split` must be a **list containing exactly one column name** (e.g., `["night"]`).
- `weights` must be a **list of label names** (subset or full).
- `other` must **not** contain any name from `features`, `time`, or `labels`.
- All arrays must share the **same first dimension** `T` (time).


## Example

```json
{
  "features": ["HB_1", "HB_2"],
  "time": ["time"],
  "labels": ["majority"],
  "split": ["night"],
  "weights": ["majority"],
  "other": ["night", "onset", "duration", "begsample", "endsample", "offset", "ai_psg", "HB_IMU_1", "HB_IMU_2", "HB_IMU_3", "HB_IMU_4", "HB_IMU_5", "HB_IMU_6", "HB_PULSE"],
  "columns": ["HB_1", "HB_2", "time", "majority", "night", "onset", "duration", "begsample", "endsample", "offset", "ai_psg", "HB_IMU_1", "HB_IMU_2", "HB_IMU_3", "HB_IMU_4", "HB_IMU_5", "HB_IMU_6", "HB_PULSE"]
}