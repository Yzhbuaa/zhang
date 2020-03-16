import numpy as np
from utils.timer import timer


def v(p, q, h):

    return np.array([
        h[0, p] * h[0, q],
        h[0, p] * h[1, q] + h[1, p] * h[0, q],
        h[1, p] * h[1, q],
        h[2, p] * h[0, q] + h[0, p] * h[2, q],
        h[2, p] * h[1, q] + h[1, p] * h[2, q],
        h[2, p] * h[2, q]
    ])


def get_camera_intrinsics(homographies):

    h_count = len(homographies)

    vec = []

    for i in range(0, h_count):
        h = np.reshape(homographies[i], (3, 3))

        vec.append(v(0, 1, h))
        vec.append(v(0, 0, h) - v(1, 1, h))

    vec = np.array(vec)

    u, s, vh = np.linalg.svd(vec)  # TODO:: mathmatic

    b = vh[-1]  # the last row of vh, the last column of v

    v0 = (b[1] * b[3] - b[0] * b[4])/(b[0] * b[2] - b[1]**2)
    lamb = b[5] - (b[3]**2 + v0 * (b[1] * b[3] - b[0] * b[4]))/b[0]
    alpha = np.sqrt(lamb/b[0])
    beta = np.sqrt(lamb * b[0] / (b[0] * b[2] - b[1]**2))
    gamma = -b[1] * alpha**2 * beta / lamb
    u0 = gamma * v0 / alpha - b[3] * alpha**2 /lamb

    return np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,      1]
    ])
