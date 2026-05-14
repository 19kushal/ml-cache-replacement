# from collections import deque

# class FIFOCache:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.cache = set()
#         self.queue = deque()

#     def access(self, key):

#         # HIT
#         if key in self.cache:
#             return True

#         # MISS
#         if len(self.cache) >= self.capacity:
#             old = self.queue.popleft()
#             self.cache.remove(old)

#         self.cache.add(key)
#         self.queue.append(key)

#         return False

from collections import OrderedDict


class FIFOCache:

    def __init__(self, capacity):

        self.capacity = capacity

        self.cache = OrderedDict()

    def access(
        self,
        key,
        op=0,
        size=0,
        key_size=0
    ):

        # HIT
        if key in self.cache:

            return True

        # MISS
        if len(self.cache) >= self.capacity:

            self.cache.popitem(last=False)

        self.cache[key] = True

        return False