import yaml
import kobe
import os

def load_yaml_config(file_path):
    """Load YAML configuration file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def gather_yaml_configs(directory):
    """Gather all YAML configurations from a directory."""
    configs = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(directory, file_name)
            configs.append(load_yaml_config(file_path))
    return configs

def main():
    """Main execution function."""
    experiment_configs = gather_yaml_configs("benchmarking")
    
    kobe.run(experiments=experiment_configs)

if __name__ == "__main__":
    main()