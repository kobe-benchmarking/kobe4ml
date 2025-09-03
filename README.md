# KOBE Benchmarking Framework

KOBE is a Python framework used to benchmark algorithms against various datasets in a standardized and reproducible way. It automates:

- **data preparation** (normalization, splitting, weights),
- **model execution** (training and inference), and
- **metrics collection** (saving results in a structured format).

With KOBE, you can run multiple experiments defined in YAML files and directly compare their results.

## Setup Instructions

**1. Create a Conda Environment**

KOBE requires Python ≥ 3.9. We recommend using a Conda environment.

```bash
conda create -n kobe python=3.12
conda activate kobe
```

**2. Install Poetry**

KOBE uses Poetry to manage dependencies and execution.

```bash
pip install poetry
```

This installs Poetry inside your Conda environment. You’ll use it to install dependencies and run the benchmark command.

**3. Define Your Experiment Folder**

Your project should follow this structure:

```graphql
experiment/
├── configs/            # YAML files defining experiments
│   ├── c1.yaml         # experiment 1
│   ├── c2.yaml         # experiment 2
│   └── ...
├── src/                # Source code for your experiments
│   ├── __init__.py
│   ├── main.py         # entrypoint script
│   └── pyproject.toml  
└── README.md         
```

**4. Install Dependencies with Poetry**

Navigate into your experiment folder and run:

```bash
cd experiment
poetry install
```

This will create a virtual environment with all dependencies defined in `pyproject.toml`.

**5. Define Experiment YAMLs**

Each YAML file in `configs/` represents one benchmark experiment. For example, `c1.yaml` might look like this:

```yaml
metadata:
  parent_id: dcoss
  id: dcoss_01
  version: v1
  name: BitBrain-LSTM-Autoencoder
  description: This run applies an LSTM autoencoder architecture that learns to reconstruct the BitBrain time series dataset.   

implementation: ...

steps:
  - id: dcoss_01_01
    type: prepare

    parameters: ...
    data: ...
    metrics: ...

  - id: dcoss_01_02
    type: work

    parameters: ...
    data: ...
    metrics: ...
```

You can use these [YAML templates](https://github.com/kobe-benchmarking/kobe4ml/tree/phase1/experiment/configs) as a reference to create your own experiment configurations.

**6. Run the Benchmark**

From inside the experiment folder run:

```bash
poetry run benchmark
```

This will create a virtual environment with all dependencies defined in `pyproject.toml`.