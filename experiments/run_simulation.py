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
# cache simulation for LRU and ML-based implementation 
def run_policy(cache, trace_df):

    hits = 0
    total_requests = len(trace_df)
    
    for req in tqdm(
        trace_df.itertuples(index=False),
        total=total_requests,
        desc=f"Simulating {cache.__class__.__name__}"
    ):

        try:
            is_hit = cache.access(req.key, req.op, req.size, req.key_size)
        except TypeError:
            is_hit = cache.access(req.key)

        if is_hit:
            hits += 1

    return hits / total_requests

def main():
    # Dataset initialization
    print("Loading trace data...")
    data = pd.read_csv("data/kv-traces-2026.csv", nrows=50000)
    data['op'] = data['op'].map({'GET': 0, 'SET': 1}).fillna(0)
    data.fillna(0, inplace=True) 

    
    print("Loading model artifacts...")
    clf = joblib.load("src/models/model.pkl")
    std_scaler = joblib.load("src/models/scaler.pkl")

    
    cache_sizes = [50, 100, 250, 500, 1000]
    metrics = []

    for size in cache_sizes:
        print(f"\nEvaluating Capacity: {size}")
        
        # Baseline: LRU
        lru = LRUCache(size)
        
        # Policy 1: Static ML inference
        static_ml = DreamOnCache(size, clf, std_scaler, online_learning=False)
        
        # Policy 2: ML with online adaptation
        dynamic_clf = copy.deepcopy(clf)
        online_ml = DreamOnCache(size, dynamic_clf, std_scaler, online_learning=True)

        # Execute simulations
        lru_rate = run_policy(lru, data)
        static_rate = run_policy(static_ml, data)
        online_rate = run_policy(online_ml, data)
        
        metrics.append({
            'Capacity': size,
            'LRU_HitRate': round(lru_rate, 4),
            'ML_Static_HitRate': round(static_rate, 4),
            'ML_Online_HitRate': round(online_rate, 4)
        })

    # Results
    print("\nSimulation Result:")
    summary_df = pd.DataFrame(metrics)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv("simulation_results_ablation.csv", index=False)

if __name__ == "__main__":
    main()