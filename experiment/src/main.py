import sys
import subprocess
import importlib

def install_and_import(package: str, git_url: str = None, index_url: str = None):
    """
    Install and import a Python package.

    :param package: Name of the package to install/import.
    :param git_url: Optional git URL for pip installation.
    :param index_url: Optional extra index URL for pip installation.
    :return: The imported module.
    """
    if git_url:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", git_url]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
        if index_url:
            cmd += ["--extra-index-url", index_url]

    subprocess.check_call(cmd)

    return importlib.import_module(package)

def main():
    # Install kobe2 directly from the phase2 branch of the git repository
    kobe2 = install_and_import(
        "kobe2", 
        git_url="git+https://github.com/kobe-benchmarking/kobe4ml.git@phase2#egg=kobe2&subdirectory=kobe2"
    )

    configs = kobe2.gather_configs(dir='configs')
    kobe2.benchmark(configs, dir='static')

if __name__ == "__main__":
    main()