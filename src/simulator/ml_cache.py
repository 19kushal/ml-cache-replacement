from collections import OrderedDict
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class DreamOnCache:
    def __init__(self, capacity, model, scaler, online_learning=True):
        self.capacity = capacity
        self.storage = OrderedDict()
        self.clf = model
        self.scaler = scaler
        self.online_updates = online_learning

        # state tracking
        self.curr_step = 0
        self.last_access_map = {}
        self.freq_map = {}
        
        # sliding window
        self.access_window = []
        self.window_freq = {} 
        
        # buffering
        self.X_buffer = []
        self.y_buffer = []
        self.min_batch = 500

    def _extract_features(self, key, op_code, val_size, key_size):
        recency = self.curr_step - self.last_access_map.get(key, self.curr_step)
        recency = max(0, recency) 
        
        total_freq = self.freq_map.get(key, 0)
        recent_freq = self.window_freq.get(key, 0)

        # feature alignment
        return [recency, total_freq, op_code, val_size, key_size, recent_freq]
    
    def access(self, key, op=0, size=0, key_size=0):
        self.curr_step += 1

        # global and window frequency
        self.freq_map[key] = self.freq_map.get(key, 0) + 1
        self.access_window.append(key)
        self.window_freq[key] = self.window_freq.get(key, 0) + 1
        
        # cwindow size
        if len(self.access_window) > 50:
            stale_key = self.access_window.pop(0)
            self.window_freq[stale_key] -= 1
            if self.window_freq[stale_key] <= 0:
                del self.window_freq[stale_key]

        # Online/adaptive training
        sample_x = self._extract_features(key, op, size, key_size)
        sample_y = 1 if self.freq_map[key] > 1 else 0

        self.X_buffer.append(sample_x)
        self.y_buffer.append(sample_y)

        if self.online_updates and len(self.X_buffer) >= self.min_batch:
            batch_data = np.array(self.X_buffer, dtype=np.float64)
            batch_data[:, [0, 1, 3, 5]] = np.log1p(batch_data[:, [0, 1, 3, 5]])
            
            x_scaled = self.scaler.transform(batch_data)
            self.clf.partial_fit(x_scaled, self.y_buffer, classes=np.array([0, 1]))
            
            self.X_buffer.clear()
            self.y_buffer.clear()

        # cache hit
        if key in self.storage:
            self.storage.move_to_end(key)
            self.last_access_map[key] = self.curr_step
            self.storage[key] = {'op': op, 'size': size, 'key_size': key_size}
            return True

        # Ccache miss
        if len(self.storage) >= self.capacity:
            keys_list = list(self.storage.keys())
            
            # Sampling
            pool_size = max(30, int(len(keys_list) * 0.10))
            pool_size = min(pool_size, len(keys_list))
            
            indices = np.random.choice(len(keys_list), pool_size, replace=False)
            candidates = [keys_list[i] for i in indices]

            # feature extraction
            feat_list = []
            for k in candidates:
                meta = self.storage[k]
                feat_list.append(self._extract_features(k, meta['op'], meta['size'], meta['key_size']))
            
            X_infer = np.array(feat_list, dtype=np.float64)
            X_infer[:, [0, 1, 3, 5]] = np.log1p(X_infer[:, [0, 1, 3, 5]])
            X_infer_scaled = self.scaler.transform(X_infer)

            # Batch inference for reuse probability
            p_reuse = self.clf.predict_proba(X_infer_scaled)[:, 1]

            candidate_scores = {}
            for i, k in enumerate(candidates):
                staleness = (self.curr_step - self.last_access_map.get(k, self.curr_step)) / self.capacity
                staleness_norm = min(staleness, 1.0)

                # Hybrid scoring 
                candidate_scores[k] = (0.9 * p_reuse[i]) - (0.1 * staleness_norm)

            # Entry eviction
            victim = min(candidate_scores, key=candidate_scores.get)
            del self.storage[victim]

        # New entry
        self.storage[key] = {'op': op, 'size': size, 'key_size': key_size}
        self.last_access_map[key] = self.curr_step

        return False