import os
import sklearn_loader as sl

logger = sl.get_logger(level='INFO')

def main():
    """
    Main function to create sklearn loaders from the Bitbrain dataset, suitable for machine learning tasks.
    """
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    
    name = 'bitbrain'
    dir = sl.get_dir(root, 'datasets', 'bitbrain_conv')
    dss = {'train': None, 'test': None}

    logger.info("Shifting labels in the entire dataset.")
    sl.shift_labels(dir, name=name)

    logger.info("Splitting data into train, test.")
    sl.split_data(dir=dir, 
                  name=name, 
                  train_size=0.7,
                  test_size=0.3)
    
    weights = sl.extract_weights(dir, name=name)
    logger.info(f"Training data class weights:\n{weights}")

    stats = sl.get_stats(dir, name=name)
    logger.info(f"Calculated statistics from training data.")

    for process in ['train', 'val', 'test']:
        logger.info(f"Normalizing {process} data with standard normalization.")
        sl.standard_normalize(dir=dir,
                              name=name,
                              process=process,
                              include=['HB_1', 'HB_2', 'time'],
                              stats=stats)
        
        logger.info(f"Normalizing {process} data with robust normalization.")
        sl.robust_normalize(dir=dir,
                            name=name,
                            process=process,
                            include=['HB_1', 'HB_2', 'time'],
                            stats=stats)
        
        logger.info(f"Creating dataset for {process} data.")
        dss[process] = sl.create_dataframe(dir=dir, 
                                           name=f'bitbrain-{process}-std-norm')

if __name__ == "__main__":
    main()