import numpy as np

from src.inference.motion import GlobalMotionCompensator, compensate_bbox


def test_compensate_bbox_applies_affine_translation() -> None:
    bbox = (30.0, 20.0, 50.0, 60.0)
    matrix = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, -5.0]], dtype=np.float32)

    compensated = compensate_bbox(bbox, matrix)

    assert compensated == (50.0, 15.0, 70.0, 55.0)


def test_global_motion_compensator_estimates_camera_translation() -> None:
    compensator = GlobalMotionCompensator(
        enabled=True,
        static_camera_assumption=False,
        method="affine",
        max_corners=200,
        min_matches=4,
        smoothing_factor=0.0,
    )

    first = np.zeros((120, 160, 3), dtype=np.uint8)
    second = np.zeros((120, 160, 3), dtype=np.uint8)
    for x, y in ((20, 20), (80, 20), (20, 80), (80, 80), (50, 50)):
        first[y - 3 : y + 3, x - 3 : x + 3] = 255
        second[y - 3 : y + 3, x + 7 - 3 : x + 7 + 3] = 255

    initial = compensator.update(first)
    motion = compensator.update(second)

    assert initial.estimated is False
    assert motion.estimated is True
    assert motion.match_count >= 4
    assert abs(float(motion.matrix[0, 2]) + 7.0) < 1.5
    assert abs(float(motion.matrix[1, 2])) < 1.5
