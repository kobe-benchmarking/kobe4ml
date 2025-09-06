from . import utils
from . import tabular as tl
from . import shared as sh

logger = utils.get_logger(level='INFO')

def main(dir, name, process, train_size, infer_size, norm_include, time_include, shifted, splitted, weighted, analyzed, normalized, weights_from, stats_from):
    """
    Main function to create sklearn loaders from the Bitbrain dataset, suitable for machine learning tasks.
    """
    dls = {}
    stats = None
    process_map = {"prepare": ["train", "val"],
                   "work": ["infer"]}

    logger.info("Shifting labels in the entire dataset.")
    sh.shift_labels(dir=dir,
                    name=name,
                    done=shifted)

    logger.info("Splitting data into train, val, infer.")
    sh.split_data(dir=dir, 
                  name=name, 
                  train_size=train_size, 
                  infer_size=infer_size,
                  done=splitted)

    for p in process_map.get(process, []):
        if weights_from == p:
            logger.info(f"Calculating class weights for {p} data.")
            sh.extract_weights(dir=dir,
                            name=name,
                            process=p,
                            done=weighted)

        if stats_from == p:
            logger.info(f"Calculating statistics for {p} data.")
            stats = tl.get_stats(dir=dir,
                                name=name,
                                process=p,
                                done=analyzed)

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
        
        logger.info(f"Creating dataset for {p} data.")
        dls[p] = sh.create_dataset(dir=dir,
                                   name=f'{name}-{p}-rbst-norm',
                                   time_include=time_include)
    
    return tuple(dls[p] for p in process_map[process])

if __name__ == "__main__":
    main()