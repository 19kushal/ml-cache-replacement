import pandas as pd
import joblib
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

from src.simulator.lru import LRUCache
from src.simulator.fifo import FIFOCache
from src.simulator.lfu import LFUCache
from src.simulator.ml_cache import MLCache

np.random.seed(42)

def run_policy(cache, df):
    hits = 0
    # Using tqdm to show progress bars for the simulation
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc=f"Running {cache.__class__.__name__}"
    ):
        # We use try/except because LRU/FIFO might not take size/op arguments in their original implementations
        try:
            hit = cache.access(row.key, row.op, row.size, row.key_size)
        except TypeError:
            hit = cache.access(row.key)

        if hit:
            hits += 1

    return hits / len(df)

def main():
    print("Loading trace data for simulation...")
    # We load a decent chunk of data to let the online learning kick in
    df = pd.read_csv("data/kv-traces-2026.csv", nrows=50000)

    # Encode operation
    # df['op'] = df['op'].map({'GET': 0, 'SET': 1})
    # Encode operation and handle any unexpected operation types or missing data
    df['op'] = df['op'].map({'GET': 0, 'SET': 1}).fillna(0)
    
    # Catch any other missing values in size or key_size
    df.fillna(0, inplace=True)

    print("Loading trained ML model and scaler...")
    model = joblib.load("src/models/model.pkl")
    scaler = joblib.load("src/models/scaler.pkl")

    # --- THE SPECTRUM EVALUATION ---
    # We test tiny caches (where ML might fail) to large caches (where ML should win)
    capacities = [50, 100, 500, 1000]
    
    final_results = []

    for capacity in capacities:
        print(f"\n=========================================")
        print(f" EVALUATING CACHE CAPACITY: {capacity}")
        print(f"=========================================")

        # Initialize all caches
        lru_cache = LRUCache(capacity)
        # fifo_cache = FIFOCache(capacity)
        # lfu_cache = LFUCache(capacity)
        
        # Initialize our custom Size-Aware Online ML Cache
        ml_cache = MLCache(capacity, model, scaler)

        # Run simulations
        lru_hit = run_policy(lru_cache, df)
        # fifo_hit = run_policy(fifo_cache, df)
        # lfu_hit = run_policy(lfu_cache, df)
        ml_hit = run_policy(ml_cache, df)
        
        # Log results
        final_results.append({
            'Capacity': capacity,
            'LRU': round(lru_hit, 4),
            # 'FIFO': round(fifo_hit, 4),
            # 'LFU': round(lfu_hit, 4),
            'ML_Cache': round(ml_hit, 4)
        })

    # --- PRINT FINAL SUMMARY TABLE ---
    print("\n" + "="*50)
    print(" FINAL SIMULATION RESULTS ")
    print("="*50)
    
    results_df = pd.DataFrame(final_results)
    print(results_df.to_string(index=False))
    
    # Save results to a CSV for your final report
    results_df.to_csv("simulation_results.csv", index=False)
    print("\nResults saved to 'simulation_results.csv' for your report.")

if __name__ == "__main__":
    main()