# # from collections import OrderedDict

# # class LRUCache:
# #     def __init__(self, capacity):
# #         self.cache = OrderedDict()
# #         self.capacity = capacity

# #     def access(self, key):
# #         if key in self.cache:
# #             self.cache.move_to_end(key)
# #             return True
# #         else:
# #             if len(self.cache) >= self.capacity:
# #                 self.cache.popitem(last=False)
# #             self.cache[key] = True
# #             return False

# # from collections import OrderedDict

# # class LRUCache:
# #     def __init__(self, capacity):
# #         self.cache = OrderedDict()
# #         self.capacity = capacity

# #     def access(self, key):
# #         if key in self.cache:
# #             self.cache.move_to_end(key)
# #             return True
# #         else:
# #             if len(self.cache) >= self.capacity:
# #                 self.cache.popitem(last=False)
# #             self.cache[key] = True
# #             return False

# from collections import OrderedDict

# class LRUCache:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.cache = OrderedDict()

#     def access(self, key):

#         # HIT
#         if key in self.cache:
#             self.cache.move_to_end(key)
#             return True

#         # MISS
#         if len(self.cache) >= self.capacity:
#             self.cache.popitem(last=False)

#         self.cache[key] = True

#         return False

from collections import OrderedDict


class LRUCache:

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

            self.cache.move_to_end(key)

            return True

        # MISS
        if len(self.cache) >= self.capacity:

            self.cache.popitem(last=False)

        self.cache[key] = True

        return False