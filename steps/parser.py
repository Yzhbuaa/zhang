import numpy as np


def parse_data(base_path="data/corners_", ext=".dat"):

    observed = []
    for i in range(1, 6):  # i=1,2,3,4,5, i.e. there are five images in total(corners_i.dat)
        observed.append(np.loadtxt(base_path + str(i) + ext).reshape((256, 2)))

    return {
        'real': np.loadtxt(base_path + "real" + ext).reshape((256, 2)),  # world coordinates unit.
        'observed': observed  # image pixel unit.
    }
