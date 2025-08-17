import kobe2

logger = kobe2.get_logger(level='INFO')

def main():
    configs = kobe2.gather_configs(dir='configs')
    
    kobe2.benchmark(configs, dir='static')

if __name__ == "__main__":
    main()