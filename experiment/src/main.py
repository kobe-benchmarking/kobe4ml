import sys
import subprocess
import importlib

def install_and_import(package: str, index_url: str = None):
    """
    Install and import a Python package.

    :param package: Name of the package to install/import.
    :param index_url: Optional extra index URL for pip installation.
    :return: The imported module.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
    if index_url:
        cmd += ["--extra-index-url", index_url]

    subprocess.check_call(cmd)

    return importlib.import_module(package)

def main():
    kobe2 = install_and_import("kobe2", index_url='https://kobe-benchmarking.github.io/kobe4ml/')

    configs = kobe2.gather_configs(dir='configs')
    kobe2.benchmark(configs, dir='static')

if __name__ == "__main__":
    main()