import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch_loader",
    version="0.30",
    author="Natalia Koliou",
    author_email="nataliakoliou@iit.demokritos.gr",
    description="Torch Loader",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kobe-benchmarking/kobe4ml/loaders/torch_loader",
    packages=setuptools.find_packages(),
    python_requires=">=3.9, <3.14",
    install_requires=[
        "pandas>=2.1,<3.0",
        "numpy>=1.26,<2.0",
        "torch>=2.0.1,<3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
