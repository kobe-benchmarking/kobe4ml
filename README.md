# Repo Structure Overview

```python
kobe4ml/
│
├── algorithms/           
│   ├── lstm_ae/       
│   │   ├── lstm_ae/     
│   │   │   ├── __init__.py  
│   │   │   ├── model.py       
│   │   │   ├── train.py 
│   │   │   ├── test.py   
│   │   │── pyproject.toml 
│   ├── conv_lstm_ae/       
│   │   ├── conv_lstm_ae/     
│   │   │   ├── __init__.py  
│   │   │   ├── model.py       
│   │   │   ├── train.py 
│   │   │   ├── test.py   
│   │   │── pyproject.toml 
│
├── experiment/         
│   ├── src/              
│   │   ├── __init__.py   
│   │   ├── main.py     
│   │   ├── utils.py     
│   │   ├── loader.py     
│   │   ├── model.py   
│   ├── configs/         
│   ├── pyproject.toml    
│
├── kobe2/         
│   ├── kobe2/              
│   │   ├── __init__.py   
│   │   ├── main.py    
│   ├── pyproject.toml    
```

# Build and Upload Packages

### 1. Build the Package

Navigate to the package directory and build the wheel:

```bash
cd algorithms/lstm_ae
poetry build
```

### 2. Upload to S3
Once the wheel file (.whl) is created inside the dist/ folder, upload it to S3:

```bash
aws s3 cp dist/lstm_ae-0.1-py3-none-any.whl s3://manolo-data/algorithms/lstm_ae-0.1-py3-none-any.whl
```