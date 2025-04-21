# Repo Structure Overview

```python
kobe4ml/
│
├── algorithms/           
│   ├── lstm_ae/       
│   │   ├── lstm_ae/     
│   │   │   ├── __init__.py  
│   │   │   ├── model.py   
│   │   │   ├── loader.py     
│   │   │   ├── train.py 
│   │   │   ├── test.py   
│   │   │   ├── utils.py    
│   │   │── pyproject.toml 
│   │   │── README.md
│   ├── conv_lstm_ae/       
│   │   ├── conv_lstm_ae/     
│   │   │   ├── __init__.py  
│   │   │   ├── model.py     
│   │   │   ├── loader.py       
│   │   │   ├── train.py 
│   │   │   ├── test.py   
│   │   │   ├── utils.py     
│   │   │── pyproject.toml 
│   │   │── README.md
│   ├── rf/       
│   │   ├── rf/     
│   │   │   ├── __init__.py     
│   │   │   ├── train.py 
│   │   │   ├── test.py   
│   │   │   ├── utils.py     
│   │   │── pyproject.toml 
│   │   │── README.md
│   ├── svm/       
│   │   ├── svm/     
│   │   │   ├── __init__.py     
│   │   │   ├── train.py 
│   │   │   ├── test.py   
│   │   │   ├── utils.py     
│   │   │── pyproject.toml 
│   │   │── README.md
│   ├── ...   
│
├── experiment/         
│   ├── src/              
│   │   ├── __init__.py   
│   │   ├── main.py     
│   │   ├── utils.py     
│   ├── configs/      
│   │   ├── c1.yaml 
│   │   ├── c2.yaml    
│   │   ├── c3.yaml    
│   │   ├── c4.yaml  
│   │   ├── ...
│   │   
│   ├── static/ 
│   │   ├── kn7fej4o
│   │   │   ├── results.csv
│   │   ├── ae2jrt8m  
│   │   │   ├── results.csv
│   │   ├── ...
│   │   
│   ├── pyproject.toml 
│   │── README.md   
│
├── kobe2/         
│   ├── kobe2/              
│   │   ├── __init__.py   
│   │   ├── main.py    
│   │   ├── utils.py   
│   ├── pyproject.toml   
│   │── README.md    
│
├── loaders/         
│   ├── bitbrain-torch-loader/    
│   │   ├── bitbrain-torch-loader/           
│   │   │   ├── __init__.py   
│   │   │   ├── main.py    
│   │   │   ├── utils.py   
│   │   ├── pyproject.toml   
│   │   │── README.md    
│   ├── bitbrain-trad-loader/    
│   │   ├── bitbrain-trad-loader/           
│   │   │   ├── __init__.py   
│   │   │   ├── main.py    
│   │   │   ├── utils.py   
│   │   ├── pyproject.toml   
│   │   │── README.md   
│   ├── ... 
│   
│── README.md    
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