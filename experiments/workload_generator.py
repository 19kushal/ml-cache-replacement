import random


def stable_locality_workload(length=20000):

    workload = []

    hot_keys = list(range(100))
    cold_keys = list(range(100, 1000))

    for _ in range(length):

        if random.random() < 0.8:
            workload.append(random.choice(hot_keys))
        else:
            workload.append(random.choice(cold_keys))

    return workload

def shifting_locality_workload(length=20000):

    workload = []

    phase1_hot = list(range(100))
    phase2_hot = list(range(500, 600))

    for i in range(length):

        # first half
        if i < length // 2:

            if random.random() < 0.8:
                workload.append(random.choice(phase1_hot))
            else:
                workload.append(random.randint(100, 1000))

        # second half
        else:

            if random.random() < 0.8:
                workload.append(random.choice(phase2_hot))
            else:
                workload.append(random.randint(100, 1000))

    return workload

def bursty_workload(length=20000):

    workload = []

    for i in range(length):

        # periodic bursts
        if (i // 1000) % 2 == 0:

            burst_keys = list(range(50))

            workload.append(random.choice(burst_keys))

        else:

            normal_keys = list(range(200, 1000))

            workload.append(random.choice(normal_keys))

    return workload

