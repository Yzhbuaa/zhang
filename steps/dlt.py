import numpy as np
from scipy import optimize as opt
from utils.timer import timer


def get_normalization_matrix(points):
    end = timer()
    #  TODO::add infinite points judgement

    u_x = points[:, 0].mean()  # Centroid
    u_y = points[:, 1].mean()

    x_shifted = points[:, 0] - u_x  # shift origin to centroid
    y_shifted = points[:, 1] - u_y

    average_distance_from_origin = np.sqrt(x_shifted**2+y_shifted**2).mean()

    scale = np.sqrt(2)/average_distance_from_origin

    end("get_normalization_matrix")

    return np.array([
        [scale,   0,   -scale * u_x],
        [0,   scale,   -scale * u_y],
        [0,       0,              1]
    ])


def estimate_homography(real, observed):
    end = timer()

    real_normalization_matrix = get_normalization_matrix(real)
    observed_normalization_matrix = get_normalization_matrix(observed)

    l = []  # using list rather than numpy.array to efficiently construct the parameter matrix step by step.

    for j in range(0, int(real.size / 2)):  # for each point in image j = 0,1,...,255
        homogeneous_real = np.array([
            real[j][0],
            real[j][1],
            1
        ])

        homogeneous_observed = np.array([
            observed[j][0],
            observed[j][1],
            1
        ])

        normalized_homogeneous_real = np.dot(real_normalization_matrix, homogeneous_real)

        normalized_homogeneous_observed = np.dot(observed_normalization_matrix, homogeneous_observed)

        l.append(np.array([
            normalized_homogeneous_real.item(0), normalized_homogeneous_real.item(1), 1,
            0, 0, 0,
            -normalized_homogeneous_real.item(0)*normalized_homogeneous_observed.item(0), -normalized_homogeneous_real.item(1)*normalized_homogeneous_observed.item(0), -normalized_homogeneous_observed.item(0)
        ]))

        l.append(np.array([
            0, 0, 0,
            normalized_homogeneous_real.item(0), normalized_homogeneous_real.item(1), 1,
            -normalized_homogeneous_real.item(0)*normalized_homogeneous_observed.item(1), -normalized_homogeneous_real.item(1)*normalized_homogeneous_observed.item(1), -normalized_homogeneous_observed.item(1)
        ]))

    u, s, vh = np.linalg.svd(np.array(l))

    x_t = vh[-1]

    h = x_t.reshape(3, 3)

    denormalized_h = np.linalg.inv(observed_normalization_matrix) @ h @ real_normalization_matrix

    end("estimate_homography")
    return denormalized_h / denormalized_h[-1, -1]


# 返回估计坐标与真实坐标偏差
def difference_value_calculate(estimated_homography, observed, real):
    estimated_coordinates = []
    for i in range(len(real)):
        single_real = np.array([real[i, 0], real[i, 1], 1])
        estimated_coordinate = np.dot(estimated_homography.reshape(3, 3), single_real)
        estimated_coordinate /= estimated_coordinate[-1]
        estimated_coordinates.append(estimated_coordinate[:2])

    aa = np.array(estimated_coordinates).reshape(-1)
    difference_value = (observed.reshape(-1) - np.array(estimated_coordinates).reshape(-1))

    return difference_value


# 返回对应jacobian矩阵
def jacobian(homography_initial_guess, observed_coordinates, real_coordinates):
    J = []
    for i in range(len(real_coordinates)):
        sx = homography_initial_guess[0] * real_coordinates[i][0] + homography_initial_guess[1] * real_coordinates[i][1] + homography_initial_guess[2]
        sy = homography_initial_guess[3] * real_coordinates[i][0] + homography_initial_guess[4] * real_coordinates[i][1] + homography_initial_guess[5]
        w = homography_initial_guess[6] * real_coordinates[i][0] + homography_initial_guess[7] * real_coordinates[i][1] + homography_initial_guess[8]
        w2 = w * w

        J.append(np.array([real_coordinates[i][0] / w, real_coordinates[i][1] / w, 1 / w,
                           0, 0, 0,
                           -sx * real_coordinates[i][0] / w2, -sx * real_coordinates[i][1] / w2, -sx / w2]))

        J.append(np.array([0, 0, 0,
                           real_coordinates[i][0] / w, real_coordinates[i][1] / w, 1 / w,
                           -sy * real_coordinates[i][0] / w2, -sy * real_coordinates[i][1] / w2, -sy / w2]))

    return np.array(J)


# 利用Levenberg Marquart算法微调H
def refine_homography(real, observed, homography_initial_guess):
    homography_initial_guess = np.array(homography_initial_guess)
    refined_homography = opt.leastsq(difference_value_calculate,
                                     homography_initial_guess,
                                     Dfun=jacobian,
                                     args=(observed, real))[0]

    refined_homography /= np.array(refined_homography[-1])
    return refined_homography


def compute_homography(data):
    end = timer()  # records the current time which will further be used to calculate the total time of calculation.

    real = data['real']

    refined_homographies = []

    for i in range(0, len(data['observed'])):  # for each image in observed 5 images
        observed = data['observed'][i]
        initial_guess = estimate_homography(real, observed)

        end = timer()
        refined_homography = refine_homography(real,observed,initial_guess)
        end("refine_homography")

        refined_homographies.append(refined_homography)

    end("compute_homography")
    return np.array(refined_homographies)


