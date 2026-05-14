# from collections import OrderedDict
# import numpy as np
# import pandas as pd
# import warnings

# # Suppress sklearn warnings about feature names during online learning
# warnings.filterwarnings("ignore", category=UserWarning)

# class MLCache:
#     def __init__(self, capacity, model, scaler):
#         self.capacity = capacity
#         self.cache = OrderedDict()
#         self.model = model
#         self.scaler = scaler

#         self.time = 0
#         self.last_seen = {}
#         self.frequency = {}
        
#         # Sliding window for recent frequency
#         self.history = []
        
#         # Online Learning Mini-Batch Queue
#         self.training_batch_X = []
#         self.training_batch_y = []
#         self.batch_size = 500  # Update the model every 500 accesses

#     def _get_features(self, key, op, size, key_size):
#         # 1. Recency
#         if key in self.last_seen:
#             recency = self.time - self.last_seen[key]
#         else:
#             recency = -1

#         # 2. Frequency
#         freq = self.frequency.get(key, 0)

#         # 3. Recent Frequency (sliding window)
#         recent = self.history.count(key)

#         # Build feature dictionary to match training exactly
#         features = {
#             "recency": recency,
#             "frequency": freq,
#             "op": op,
#             "size": size,
#             "key_size": key_size,
#             "recent_freq": recent
#         }

#         df_features = pd.DataFrame([features])
        
#         # Defensive check: ensure no NaNs sneak into the prediction vector
#         df_features.fillna(0, inplace=True)

#         # Apply the EXACT SAME Log-Transforms we used in training

#         # Apply the EXACT SAME Log-Transforms we used in training
#         skewed_features = ['recency', 'frequency', 'size', 'recent_freq']
#         for col in skewed_features:
#             df_features[col] = np.log1p(df_features[col].clip(lower=0))

#         # Scale and return
#         return self.scaler.transform(df_features)
    
#     def access(self, key, op=0, size=0, key_size=0):
#         self.time += 1

#         # --- UPDATE TRACKING STATS ---
#         self.frequency[key] = self.frequency.get(key, 0) + 1
        
#         self.history.append(key)
#         if len(self.history) > 50:  # Keep window fixed at 50
#             self.history.pop(0)

#         # --- ONLINE LEARNING BATCHING ---
#         # Generate features for the current access
#         X_current = self._get_features(key, op, size, key_size)[0]
        
#         # Define the online label: If we've seen it more than once, it's a "reuse"
#         y_current = 1 if self.frequency[key] > 1 else 0

#         self.training_batch_X.append(X_current)
#         self.training_batch_y.append(y_current)

#         # If batch is full, adapt the model to new workload patterns
#         if len(self.training_batch_X) >= self.batch_size:
#             self.model.partial_fit(
#                 self.training_batch_X, 
#                 self.training_batch_y, 
#                 classes=np.array([0, 1])
#             )
#             # Flush the queue
#             self.training_batch_X = []
#             self.training_batch_y = []

#         # --- CACHE HIT ---
#         if key in self.cache:
#             self.cache.move_to_end(key)
#             self.last_seen[key] = self.time
#             # Update stored size metadata just in case
#             self.cache[key] = {'op': op, 'size': size, 'key_size': key_size}
#             return True

#         # --- CACHE MISS & EVICTION ---
#         # if len(self.cache) >= self.capacity:
#         #     cache_keys = list(self.cache.keys())
            
#         #     # Sample candidates to save computation time (standard in ML caches)
#         #     candidate_size = min(30, len(cache_keys))
#         #     indices = np.random.choice(len(cache_keys), candidate_size, replace=False)
#         #     candidates = [cache_keys[i] for i in indices]

#         #     scores = {}

#         #     for k in candidates:
#         #         meta = self.cache[k]
#         #         features = self._get_features(k, meta['op'], meta['size'], meta['key_size'])
                
#         #         # Get Probability of Reuse (Class 1)
#         #         prob_reuse = self.model.predict_proba(features)[0][1]

#         #         # CUSTOM LOGIC: Size-Aware Utility Score
#         #         # High prob_reuse = High Score (Keep it)
#         #         # High size = Lower Score (Evict it)
#         #         utility_score = prob_reuse / (meta['size'] + 1)

#         #         scores[k] = utility_score

#         #     # Evict the item with the ABSOLUTE LOWEST utility score
#         #     evict_key = min(scores, key=scores.get)
#         #     del self.cache[evict_key]
#         # --- CACHE MISS & EVICTION ---
#         if len(self.cache) >= self.capacity:
#             cache_keys = list(self.cache.keys())
            
#             # DYNAMIC SAMPLING: Sample 10% of the cache, but at least 30 items
#             candidate_size = max(30, int(len(cache_keys) * 0.10))
#             candidate_size = min(candidate_size, len(cache_keys)) # Safety check
            
#             indices = np.random.choice(len(cache_keys), candidate_size, replace=False)
#             candidates = [cache_keys[i] for i in indices]

#             scores = {}

#             for k in candidates:
#                 meta = self.cache[k]
#                 features = self._get_features(k, meta['op'], meta['size'], meta['key_size'])
                
#                 # Get Probability of Reuse (Class 1)
#                 prob_reuse = self.model.predict_proba(features)[0][1]

#                 # HYBRID SCORE: ML Probability + Recency Safety Net
#                 # Calculate how "stale" the item is relative to the cache capacity
#                 recency_staleness = (self.time - self.last_seen.get(k, self.time)) / self.capacity
#                 normalized_staleness = min(recency_staleness, 1.0)

#                 # We heavily weight the ML probability (80%), but penalize items that have been sitting 
#                 # untouched for a very long time (20%) to prevent "dead" items from clogging the cache.
#                 hybrid_score = (0.9 * prob_reuse) - (0.1 * normalized_staleness)

#                 scores[k] = hybrid_score

#             # Evict the item with the ABSOLUTE LOWEST hybrid score
#             evict_key = min(scores, key=scores.get)
#             del self.cache[evict_key]

#         # --- INSERT NEW ITEM ---
#         self.cache[key] = {'op': op, 'size': size, 'key_size': key_size}
#         self.last_seen[key] = self.time

#         return False


from collections import OrderedDict
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class DreamOnCache:
    def __init__(self, capacity, model, scaler, online_learning=True):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.model = model
        self.scaler = scaler
        
        # New toggle flag
        self.online_learning = online_learning

        self.time = 0
        self.last_seen = {}
        self.frequency = {}
        
        self.history = []
        self.recent_freq_map = {} 
        
        self.training_batch_X = []
        self.training_batch_y = []
        self.batch_size = 500

    def _get_raw_features(self, key, op, size, key_size):
        """Returns raw features as a list without Pandas overhead."""
        recency = self.time - self.last_seen.get(key, self.time)
        recency = max(0, recency) # Clip to 0 to prevent log1p issues
        
        freq = self.frequency.get(key, 0)
        recent = self.recent_freq_map.get(key, 0)

        # Order must match training: ['recency', 'frequency', 'op', 'size', 'key_size', 'recent_freq']
        return [recency, freq, op, size, key_size, recent]
    
    def access(self, key, op=0, size=0, key_size=0):
        self.time += 1

        # --- UPDATE TRACKING STATS ---
        self.frequency[key] = self.frequency.get(key, 0) + 1
        
        # O(1) Sliding Window Update
        self.history.append(key)
        self.recent_freq_map[key] = self.recent_freq_map.get(key, 0) + 1
        
        if len(self.history) > 50:
            old_key = self.history.pop(0)
            self.recent_freq_map[old_key] -= 1
            if self.recent_freq_map[old_key] <= 0:
                del self.recent_freq_map[old_key]

        # --- ONLINE LEARNING BATCHING ---
        raw_features = self._get_raw_features(key, op, size, key_size)
        y_current = 1 if self.frequency[key] > 1 else 0

        self.training_batch_X.append(raw_features)
        self.training_batch_y.append(y_current)

        # OPTIMIZATION 2: Batch scale and fit the online learning
        if self.online_learning and len(self.training_batch_X) >= self.batch_size:
            X_batch = np.array(self.training_batch_X, dtype=np.float64)
            X_batch[:, [0, 1, 3, 5]] = np.log1p(X_batch[:, [0, 1, 3, 5]])
            
            X_batch_scaled = self.scaler.transform(X_batch)
            self.model.partial_fit(X_batch_scaled, self.training_batch_y, classes=np.array([0, 1]))
            
            self.training_batch_X = []
            self.training_batch_y = []

        # --- CACHE HIT ---
        if key in self.cache:
            self.cache.move_to_end(key)
            self.last_seen[key] = self.time
            self.cache[key] = {'op': op, 'size': size, 'key_size': key_size}
            return True

        # --- CACHE MISS & EVICTION ---
        if len(self.cache) >= self.capacity:
            cache_keys = list(self.cache.keys())
            
            candidate_size = max(30, int(len(cache_keys) * 0.10))
            candidate_size = min(candidate_size, len(cache_keys))
            
            # Faster sampling
            indices = np.random.choice(len(cache_keys), candidate_size, replace=False)
            candidates = [cache_keys[i] for i in indices]

            # OPTIMIZATION 3: Batched Inference
            # Extract features for all candidates first
            candidate_features = []
            for k in candidates:
                meta = self.cache[k]
                candidate_features.append(self._get_raw_features(k, meta['op'], meta['size'], meta['key_size']))
            
            # Convert to NumPy, Log Transform, and Scale all at once
            X_candidates = np.array(candidate_features, dtype=np.float64)
            X_candidates[:, [0, 1, 3, 5]] = np.log1p(X_candidates[:, [0, 1, 3, 5]])
            X_candidates_scaled = self.scaler.transform(X_candidates)

            # Predict probabilities for all candidates in one single Matrix operation
            probs = self.model.predict_proba(X_candidates_scaled)[:, 1]

            scores = {}
            for i, k in enumerate(candidates):
                prob_reuse = probs[i]
                
                recency_staleness = (self.time - self.last_seen.get(k, self.time)) / self.capacity
                normalized_staleness = min(recency_staleness, 1.0)

                hybrid_score = (0.9 * prob_reuse) - (0.1 * normalized_staleness)
                scores[k] = hybrid_score

            evict_key = min(scores, key=scores.get)
            del self.cache[evict_key]

        # --- INSERT NEW ITEM ---
        self.cache[key] = {'op': op, 'size': size, 'key_size': key_size}
        self.last_seen[key] = self.time

        return False