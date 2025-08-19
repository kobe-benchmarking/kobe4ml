import os
from . import utils
from . import tabular as tl
from . import shared as sh

logger = utils.get_logger(level='INFO')

def main(dir, name, process, batch_size, train_size, val_size, test_size, seq_len, norm_include, full_epoch, per_epoch):
    """
    Main function to create torch loaders from the Bitbrain dataset, suitable for machine learning tasks.
    """
    dls = {'train': None, 'val': None, 'test': None}

    process_map = {"prepare": ["train", "val"],
                   "work": ["test"]}

    logger.info("Shifting labels in the entire dataset.")
    sh.shift_labels(dir, name=name)

    logger.info("Splitting data into train, val, test.")
    sh.split_data(dir=dir, 
                  name=name, 
                  train_size=train_size, 
                  val_size=val_size, 
                  test_size=test_size)
    
    weights = sh.extract_weights(dir, name=name)
    logger.info(f"Training data class weights:\n{weights}")

    stats = tl.get_stats(dir, name=name)
    logger.info(f"Calculated statistics from training data.")

    for p in process_map.get(process, []):
        logger.info(f"Normalizing {p} data with standard normalization.")
        tl.standard_normalize(dir=dir,
                              name=name,
                              process=p,
                              include=norm_include,
                              stats=stats)
        
        logger.info(f"Normalizing {p} data with robust normalization.")
        tl.robust_normalize(dir=dir,
                            name=name,
                            process=p,
                            include=norm_include,
                            stats=stats)
        
        logger.info(f"Creating TSDataset for {p} data.")
        ds = tl.TSDataset(dir=dir, 
                          name=f'{name}-{p}-rbst-norm',
                          seq_len=seq_len,
                          full_epoch=full_epoch,
                          per_epoch=per_epoch)
        
        logger.info(f"Creating dataloader for {p} data.")
        dls[p] = sh.create_dataloader(ds=ds, 
                                      batch_size=batch_size, 
                                      shuffle=[True, False, False], 
                                      num_workers=None, 
                                      drop_last=False)
    
    return dls

if __name__ == "__main__":
    main()