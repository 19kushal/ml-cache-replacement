# class LFUCache:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.cache = {}
#         self.freq = {}

#     def access(self, key):

#         # HIT
#         if key in self.cache:
#             self.freq[key] += 1
#             return True

#         # MISS
#         if len(self.cache) >= self.capacity:

#             evict_key = min(
#                 self.freq,
#                 key=self.freq.get
#             )

#             del self.cache[evict_key]
#             del self.freq[evict_key]

#         self.cache[key] = True
#         self.freq[key] = 1

#         return False

class LFUCache:

    def __init__(self, capacity):

        self.capacity = capacity

        self.cache = {}

        self.freq = {}

    def access(
        self,
        key,
        op=0,
        size=0,
        key_size=0
    ):

        # HIT
        if key in self.cache:

            self.freq[key] += 1

            return True

        # MISS
        if len(self.cache) >= self.capacity:

            lfu_key = min(
                self.freq,
                key=self.freq.get
            )

            del self.cache[lfu_key]

            del self.freq[lfu_key]

        self.cache[key] = True

        self.freq[key] = 1

        return False