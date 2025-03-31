import os
import pandas as pd

def main(calls, runs, dir, logger):
    results = []
    
    for i, call in enumerate(calls):
        try:
            logger.info(f"Running call {i}...")
            
            metrics = call()
            metrics["run"] = runs[i]

        except Exception as e:
            metrics = {"run": i, "status": "error", "error": str(e)}
        
        logger.info(f"Metrics for run {i}: {metrics}")
        results.append(metrics)

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(dir, "results.csv")
        df.to_csv(csv_path, index=False)

        logger.info(f"Results saved to {csv_path}")