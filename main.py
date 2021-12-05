

from OutlierDetection import OutlierDetect
from SpacePartition import *
from OutlierDetection import *
from SpacePartition import show_regions


if __name__ == "__main__":

    data = np.load("./seeds.npy")

    outliers = []
    patterns = []
    results = DHC(data)

    for r in results:

        p, o = OutlierDetect(r)
        patterns += p
        outliers += o
        # show_regions(p)
    
    # do something for the seed region list,  i.e., patterns
    # Note the Outliers can be used for the next iter input
