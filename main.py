from steps.parser import parse_data
from steps.dlt import compute_homography
from steps.intrinsics import get_camera_intrinsics
from steps.extrinsics import get_camera_extrinsics
from steps.distortion import estimate_lens_distortion
from utils.timer import timer
from steps.refine_all import refinall_all_param

def calibrate():
    # reads all data(points's coordinate in both image coordinate system and world coordinate system
    data = parse_data()

    end = timer()
    homographies = compute_homography(data)
    end("Homography Estimation")
    print("homographies")
    print(homographies)

    end = timer()
    intrinsics = get_camera_intrinsics(homographies)
    end("Intrinsics")

    print("intrinsics")
    print(intrinsics)

    end = timer()
    extrinsics = get_camera_extrinsics(intrinsics, homographies)
    end("Extrinsics")

    print("extrinsics")
    print(extrinsics)

    end = timer()
    distortion = estimate_lens_distortion(
        intrinsics,
        extrinsics,
        data["real"],
        data["observed"]
    )
    end("Distortion")

    end = timer()
    [refined_intrinsics_parameter, refined_distortion_parameter, refined_extrinsic_parameter] = \
        refinall_all_param(intrinsics, distortion, extrinsics, data['real'], data['observed'])

    return

calibrate()
