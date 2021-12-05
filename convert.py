# convert IPv6 str to numpy seeds.npy

import numpy as np
from IPy import IP
import sys


with open("./seeds") as f:
    arrs = []
    for ip in f.read().splitlines()[:10000]:
        arrs.append([int(x, 16)
                    for x in IP(ip).strFullsize().replace(":", "")])

    np.save("seeds.npy", np.array(arrs, dtype=np.uint8))


# with open("responsive-addresses.txt") as f:
#     arrs = []
#     ip = f.readline()[:-1]
#     num = 0
#     while num < 10000000:
#         num += 1
#         ip = f.readline()[:-1]
#         arrs.append([int(x, 16)
#                      for x in IP(ip).strFullsize().replace(":", "")])

#     np.save("seeds.npy", np.array(arrs, dtype=np.uint8))
