from . import utils
from . import tabular as tl
from . import shared as sh

logger = utils.get_logger(level='INFO')

def main(dir, name, process, train_size, val_size, test_size, seq_len, norm_include, full_epoch, per_epoch, time_include, shifted, splitted, weighted, analyzed, normalized):
    """
    Main function to create torch loaders from the Bitbrain dataset, suitable for machine learning tasks.
    """
    dls = {}
    process_map = {"prepare": ["train", "val"],
                   "work": ["test"]}

    logger.info("Shifting labels in the entire dataset.")
    sh.shift_labels(dir, name=name, done=shifted)

    logger.info("Splitting data into train, val, test.")
    sh.split_data(dir=dir, 
                  name=name, 
                  train_size=train_size, 
                  val_size=val_size, 
                  test_size=test_size,
                  done=splitted)
    
    weights = sh.extract_weights(dir, name=name, done=weighted)
    logger.info(f"Training data class weights:\n{weights}")

    stats = tl.get_stats(dir, name=name, done=analyzed)
    logger.info(f"Calculated statistics from training data.")

    for p in process_map.get(process, []):
        logger.info(f"Normalizing {p} data with standard normalization.")
        tl.standard_normalize(dir=dir,
                              name=name,
                              process=p,
                              include=norm_include,
                              stats=stats,
                              done=normalized)
        
        logger.info(f"Normalizing {p} data with robust normalization.")
        tl.robust_normalize(dir=dir,
                            name=name,
                            process=p,
                            include=norm_include,
                            stats=stats,
                            done=normalized)
        
        logger.info(f"Creating TSDataset for {p} data.")
        dls[p] = tl.TSDataset(dir=dir, 
                              name=f'{name}-{p}-rbst-norm',
                              seq_len=seq_len,
                              full_epoch=full_epoch,
                              per_epoch=per_epoch,
                              time_include=time_include)
    
    return tuple(dls[p] for p in process_map[process])

if __name__ == "__main__":
    main()