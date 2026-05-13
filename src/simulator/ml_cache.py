from collections import OrderedDict
import numpy as np

class MLCache:
    def __init__(self, capacity, model, scaler, threshold):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

        self.last_seen = {}
        self.freq = {}
        self.meta = {}        
        self.history = []     
        self.time = 0
        self.recent_freq_map = {}

        self.rng = np.random.default_rng(42)


    def _get_features(self, key):
        if key in self.last_seen:
            recency = self.time - self.last_seen[key]
        else:
            recency = 0   
        frequency = self.freq.get(key, 0)
        op, size, key_size = self.meta.get(key, (0, 0, 0))
        # recent_freq = self.history.count(key)
        recent_freq = self.recent_freq_map.get(key, 0)

        # features = np.array([[recency, frequency, op, size, key_size, recent_freq]])
        velocity = frequency / (recency + 1)

        features = np.array([[
            recency,
            frequency,
            op,
            size,
            key_size,
            recent_freq,
            velocity
        ]])

        features = np.nan_to_num(features)
        return self.scaler.transform(features)
    
    def access(self, key, op=0, size=0, key_size=0):
        self.time += 1

        # update stats
        self.freq[key] = self.freq.get(key, 0) + 1
        self.meta[key] = (op, size, key_size)

        self.history.append(key)
        self.recent_freq_map[key] = self.recent_freq_map.get(key, 0) + 1

        if len(self.history) > 50:
            old_key = self.history.pop(0)

            self.recent_freq_map[old_key] -= 1
            if self.recent_freq_map[old_key] <= 0:
                del self.recent_freq_map[old_key]

        # HIT
        if key in self.cache:
            self.cache.move_to_end(key)
            self.last_seen[key] = self.time
            return True

        # MISS
        if len(self.cache) >= self.capacity:
            # scores = {}
            # # candidates = list(self.cache.keys())[:10]
            # #   # only check 20 items
            # cache_keys = list(self.cache.keys())

            # candidate_size = min(20, len(cache_keys))

            # indices = self.rng.choice(
            #     len(cache_keys),
            #     candidate_size,
            #     replace=False
            # )

            # candidates = [cache_keys[i] for i in indices]
            # # for k in candidates:
            # #     features = self._get_features(k)
            # #     prob = self.model.predict_proba(features)[0][1]

            # #     # hybrid score (ML + recency)
            # #     # recency = self.time - self.last_seen.get(k, self.time)
            # #     # # score = (0.8 * prob) - (0.2 * (recency / self.capacity))
            # #     # normalized_recency = min(recency / 100, 1.0)
            # #     # score = (0.8 * prob) - (0.2 * normalized_recency)
            # #     score = prob

            # #     scores[k] = score


            # # evict_key = min(scores, key=scores.get)
            # eviction_candidates = {}

            # for k in candidates:

            #     features = self._get_features(k)

            #     prob = self.model.predict_proba(features)[0][1]

            #     # items below threshold are eviction candidates
            #     if prob < self.threshold:
            #         eviction_candidates[k] = prob

            #     # fallback
            #     scores[k] = prob

            # # Prefer evicting low-confidence items
            # if eviction_candidates:
            #     evict_key = min(eviction_candidates, key=eviction_candidates.get)
            # else:
            #     evict_key = min(scores, key=scores.get)
            #     del self.cache[evict_key]
            scores = {}
            eviction_candidates = {}

            cache_keys = list(self.cache.keys())

            candidate_size = min(10, len(cache_keys))

            indices = np.random.choice(
                len(cache_keys),
                candidate_size,
                replace=False
            )

            candidates = [cache_keys[i] for i in indices]

            for k in candidates:

                features = self._get_features(k)

                prob = self.model.predict_proba(features)[0][1]

                scores[k] = prob

                # low confidence items become eviction candidates
                if prob < self.threshold:
                    eviction_candidates[k] = prob

            # choose eviction victim
            if eviction_candidates:
                evict_key = min(
                    eviction_candidates,
                    key=eviction_candidates.get
                )
            else:
                evict_key = min(scores, key=scores.get)

# IMPORTANT
            del self.cache[evict_key]

        self.cache[key] = True
        self.last_seen[key] = self.time

        # if self.time % 100 == 0:

        #     X_online = np.vstack([
        #     self._get_features(key),
        #     self._get_features(key)
        #     ])

        #     y_online = np.array([0, 1])
        #     self.model.partial_fit(X_online, y_online)

        return False
    
