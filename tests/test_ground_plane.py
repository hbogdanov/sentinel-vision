from src.inference.ground_plane import GroundPlaneMapper


def test_ground_plane_mapper_projects_points_with_identity_homography() -> None:
    mapper = GroundPlaneMapper.from_correspondences(
        image_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
        world_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )

    assert mapper.project_point((5, 7)) == (5.0, 7.0)
    assert mapper.project_points([(1, 1), (2, 2)]) == [(1.0, 1.0), (2.0, 2.0)]
