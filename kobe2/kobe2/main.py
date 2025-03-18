import os
import pandas as pd

def main(calls, dir):
    results = []
    
    for i, call in enumerate(calls):
        try:
            metrics = call()
            results.append(metrics)
        except Exception as e:
            print(f"Experiment {i} failed: {e}")
            results.append({'error': str(e)})

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(dir, "results.csv")
        df.to_csv(csv_path, index=False)

        print(f"Results saved to {csv_path}")