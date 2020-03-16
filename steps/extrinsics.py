import numpy as np


def estimate_view_transform(intrinsics, homography):

    homography = homography.reshape(3, 3)
    inv_intrinsics = np.linalg.inv(intrinsics)

    h1 = homography[:, 0]
    h2 = homography[:, 1]
    h3 = homography[:, 2]

    # Calculate the 1/s constant in eq.(2)
    # Theoretically, ld1 == ld2, however, because of noise, ld1 != ld2, so uses their mean value ld3 instead.
    ld1 = 1 / np.linalg.norm(np.dot(inv_intrinsics, h1))  # Frobenius norm
    ld2 = 1 / np.linalg.norm(np.dot(inv_intrinsics, h2))
    ld3 = (ld1 + ld2) / 2

    r0 = np.array([ld1 * np.dot(inv_intrinsics, h1)]).transpose()
    r1 = np.array([ld2 * np.dot(inv_intrinsics, h2)]).transpose()
    r2 = np.cross(r0, r1, axis=0)

    t = np.array([ld3 * np.dot(inv_intrinsics, h3)]).transpose()

    r = np.concatenate((r0, r1, r2), axis=1)

    r = denoise_rotation_matrix(r)

    rt = np.concatenate((r, t), axis=1)

    return rt


def denoise_rotation_matrix(rotation_matrix):
    u, s, vh = np.linalg.svd(rotation_matrix)
    return u @ vh


def get_camera_extrinsics(intrinsics, homographies):

    extrinsics = []
    for i in range(0, len(homographies)):
        extrinsics.append(estimate_view_transform(intrinsics, homographies[i]))

    return extrinsics
