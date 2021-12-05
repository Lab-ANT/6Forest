import numpy as np
import queue




def DHC(arrs):
    # q = queue.Queue()
    q = queue.LifoQueue()
    q.put(arrs)
    regions_arrs = []
    while not q.empty():
        arrs = q.get()
        if len(arrs) < 16:
            regions_arrs.append(arrs)
            continue
        splits = maxcovering(arrs)

        for s in splits:
            q.put(arrs[s])

    return regions_arrs


def maxcovering(arrs):
    Tarrs = arrs.T
    Covering = []
    for i in range(32):
        splits = np.bincount(Tarrs[i], minlength=16)
        if np.count_nonzero(splits) == 1:
            # fixed dimension
            Covering.append(-1)
        else:
            Covering.append(np.sum(splits*(splits != 1)))
    index = np.argmax(Covering)
    splits = np.bincount(Tarrs[index], minlength=16)
    split_nibbles = np.argwhere(splits).reshape(-1)

    return [
        np.argwhere(Tarrs[index] == nibble).reshape(-1) for nibble in split_nibbles
    ]


def leftmost(arrs):
    Tarrs = arrs.T
    for i in range(32):
        splits = np.bincount(Tarrs[i], minlength=16)

        if len(splits[splits > 0]) > 1:
            split_index = i
            split_nibbles = np.where(splits != 0)[0]
            break

    return [
        np.where(Tarrs[split_index] == nibble)[0] for nibble in split_nibbles
    ]


# show the regions for test

def show_regions(arrs):
    address_space = []
    Tarrs = arrs.T
    for i in range(32):
        splits = np.bincount(Tarrs[i], minlength=16)
        # print(i, splits, np.argwhere(splits > 0)[0][0])
        if len(splits[splits > 0]) == 1:

            address_space.append(format(np.argwhere(splits > 0)[0][0], "x"))
        else:
            address_space.append("*")

    print("********Region**********")
    print("".join(address_space))
    for i in range(len(arrs)):
        print("".join([format(x, "x") for x in arrs[i]]), " ", i)
    print()
