import os
import pandas as pd

def main(calls, runs, dir):
    results = []
    
    for i, call in enumerate(calls):
        try:
            metrics = call()
            metrics["run"] = runs[i]
        except Exception as e:
            metrics = {"run": i, "status": "error", "error": str(e)}
        
        results.append(metrics)

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(dir, "results.csv")
        df.to_csv(csv_path, index=False)

        print(f"Results saved to {csv_path}")