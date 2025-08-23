import os
import torch_loader as tl

logger = tl.get_logger(level='INFO')

def main():
    """
    Main function to create torch loaders from the Bitbrain dataset, suitable for machine learning tasks.
    """
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    
    name = 'bitbrain'
    dir = tl.get_dir(root, 'datasets', 'bitbrain_conv')
    dls = {'train': None, 'val': None, 'test': None}

    logger.info("Shifting labels in the entire dataset.")
    tl.shift_labels(dir, name=name)

    logger.info("Splitting data into train, val, test.")
    tl.split_data(dir=dir, 
                  name=name, 
                  train_size=0.6, 
                  val_size=0.2, 
                  test_size=0.2)
    
    weights = tl.extract_weights(dir, name=name)
    logger.info(f"Training data class weights:\n{weights}")

    stats = tl.get_stats(dir, name=name)
    logger.info(f"Calculated statistics from training data.")

    for process in ['train', 'val', 'test']:
        logger.info(f"Normalizing {process} data with standard normalization.")
        tl.standard_normalize(dir=dir,
                              name=name,
                              process=process,
                              include=['HB_1', 'HB_2', 'time'],
                              stats=stats)
        
        logger.info(f"Normalizing {process} data with robust normalization.")
        tl.robust_normalize(dir=dir,
                            name=name,
                            process=process,
                            include=['HB_1', 'HB_2', 'time'],
                            stats=stats)
        
        logger.info(f"Creating TSDataset for {process} data.")
        ds = tl.TSDataset(dir=dir, 
                          name=f'bitbrain-{process}-std-norm',
                          seq_len=240,
                          full_epoch=7680,
                          per_epoch=True)
        
        logger.info(f"Creating dataloader for {process} data.")
        dls[process] = tl.create_dataloader(ds=ds, 
                                            batch_size=512, 
                                            shuffle=[True, False, False], 
                                            num_workers=None, 
                                            drop_last=False)

if __name__ == "__main__":
    main()