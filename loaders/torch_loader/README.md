```
[tool.poetry]
name = "torch_loader"
version = "0.30"
packages = [
    { include = "torch_loader" }
]
description = "Torch Loader"
authors = [ "Natalia Koliou <nataliakoliou@iit.demokritos.gr>",
	        "Stasinos Konstantopoulos <konstant@iit.demokritos.gr>" ]
readme = "README.md"

[tool.poetry.dependencies]

# Python version for the project
python = ">=3.9,<3.14"  

# Loading and creating dataloaders
pandas = "^2.1"         # For data manipulation and analysis
numpy = "^1.26"         # For numerical computations (e.g., tensor operations)
torch = "^2.0.1"        # For creating PyTorch dataloaders

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```