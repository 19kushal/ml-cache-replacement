import copy
import pandas as pd
import joblib
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

from src.simulator.lru import LRUCache
from src.simulator.ml_cache import DreamOnCache

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
    df = pd.read_csv("data/kv-traces-2026.csv", nrows=50000)
    df['op'] = df['op'].map({'GET': 0, 'SET': 1}).fillna(0)
    df.fillna(0, inplace=True) 

    print("Loading trained ML base model and scaler...")
    base_model = joblib.load("src/models/model.pkl")
    scaler = joblib.load("src/models/scaler.pkl")

    capacities = [50, 100, 250, 500, 1000]
    final_results = []

    for capacity in capacities:
        print(f"\n=========================================")
        print(f" EVALUATING CACHE CAPACITY: {capacity}")
        print(f"=========================================")

        lru_cache = LRUCache(capacity)
        
        # 1. Static ML Cache (No Online Learning)
        ml_cache_static = DreamOnCache(capacity, base_model, scaler, online_learning=False)
        
        # 2. Online ML Cache (Requires a deepcopy so it doesn't mutate the base model)
        online_model = copy.deepcopy(base_model)
        ml_cache_online = DreamOnCache(capacity, online_model, scaler, online_learning=True)

        lru_hit = run_policy(lru_cache, df)
        ml_static_hit = run_policy(ml_cache_static, df)
        ml_online_hit = run_policy(ml_cache_online, df)
        
        final_results.append({
            'Capacity': capacity,
            'LRU': round(lru_hit, 4),
            'ML_Static': round(ml_static_hit, 4),
            'ML_Online': round(ml_online_hit, 4)
        })

    print("\n" + "="*50)
    print(" FINAL SIMULATION RESULTS (ABLATION STUDY) ")
    print("="*50)
    
    results_df = pd.DataFrame(final_results)
    print(results_df.to_string(index=False))
    results_df.to_csv("simulation_results_ablation.csv", index=False)
if __name__ == "__main__":
    main()