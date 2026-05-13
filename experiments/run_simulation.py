# import pandas as pd
# import joblib

# from src.simulator.lru import LRUCache
# from src.simulator.ml_cache import MLCache
# # df = pd.read_csv('/Users/nimishajoshi/ml-cache-replacement/data/kv-traces-2026.csv')
# # Only load the 'key' column to save massive amounts of RAM
# df = pd.read_csv('/Users/nimishajoshi/ml-cache-replacement/data/kv-traces-2026.csv', usecols=['key'])

# model = joblib.load("src/models/model.pkl")
# scaler = joblib.load("src/models/scaler.pkl")

# def run(cache, sequence):
#     hits = 0

#     for key in sequence:
#         if cache.access(key):
#             hits += 1

#     return hits / len(sequence)


# # prepare sequence
# # sequence = df['key'].tolist()
# sequence = df['key'].tolist()[:20000]

# capacity = 1000

# # LRU
# lru_cache = LRUCache(capacity)
# lru_hit_rate = run(lru_cache, sequence)

# # ML
# ml_cache = MLCache(capacity, model, scaler)
# ml_hit_rate = run(ml_cache, sequence)

# print("LRU Hit Rate:", lru_hit_rate)
# print("ML Hit Rate:", ml_hit_rate)

import pandas as pd
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm

from src.simulator.lru import LRUCache
from src.simulator.ml_cache import MLCache

np.random.seed(42)

model = joblib.load("src/models/model.pkl")
scaler = joblib.load("src/models/scaler.pkl")

def run_lru(sequence, capacity):
    cache = LRUCache(capacity)
    hits = 0

    for key in sequence:
        if cache.access(key):
            hits += 1

    return hits / len(sequence)


def run_ml(df, capacity, model, scaler):

    # cache = MLCache(capacity, model, scaler)
    cache = MLCache(
    capacity,
    model,
    scaler,
    threshold=0.5
)

    hits = 0

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Running ML Cache"
    ):

        if cache.access(
            row.key,
            row.op,
            row.size,
            row.key_size
        ):
            hits += 1

    return hits / len(df)

def main():
    # LOAD DATA (SMALL SAMPLE)
    # df = pd.read_csv("data/cache_trace.txt", nrows=20000)
    # df = pd.read_csv('/Users/nimishajoshi/ml-cache-replacement/data/kv-traces-2026.csv')
    df = pd.read_csv("../data/kv-traces-2026.csv", nrows=20000)


    df['op'] = df['op'].map({'GET': 0, 'SET': 1})

    # LOAD MODEL
    # model = joblib.load("/Users/nimishajoshi/ml-cache-replacement/src/models/model.pkl")
    # scaler = joblib.load("/Users/nimishajoshi/ml-cache-replacement/src/models/scaler.pkl")

    # capacity = 100

    # sequence = df['key'].tolist()

    # lru_hit_rate = run_lru(sequence, capacity)
    # ml_hit_rate = run_ml(df, capacity, model, scaler)

    # print("LRU Hit Rate:", lru_hit_rate)
    # print("ML Hit Rate:", ml_hit_rate)
    capacities = [500, 1000, 1500, 2000]

    sequence = df['key'].tolist()

    for capacity in capacities:

        print(f"\nTesting Cache Capacity = {capacity}")

        lru_hit_rate = run_lru(sequence, capacity)
        ml_hit_rate = run_ml(df, capacity, model, scaler)

        print("LRU Hit Rate:", round(lru_hit_rate, 5))
        print("ML Hit Rate:", round(ml_hit_rate, 5))


if __name__ == "__main__":
    main()