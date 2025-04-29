import numpy as np
import open3d as o3d


def draw_camera(
    K, R, t, w, h,
    scale=1,
    color=[0.8, 0.2, 0.8],
    draw_axis=True,
    draw_plane=True,
):
    """Create axis, plane and pyramed geometries in Open3D format.
    Args:
        K: camera intrinsics
        R: rotation matrix (camera_to_world)
        t: translation vector (camera_to_world)
        w: image width
        h: image height
        scale: camera model scale
        color: color of the camera
        draw_axis: whether to draw axis
        draw_plane: whether to draw plane
    Returns:
        List of Open3D geometries (axis, plane and pyramid)
    """
    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 4],
        [3, 4],
        [1, 3]
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_in_world),
        lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    # return [axis, plane, line_set]
    output = [line_set]
    # axis
    if draw_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8 * scale)
        axis.transform(T)
        output.append(axis)

    if draw_plane:
        # image plane
        width = abs(points[1][0]) + abs(points[3][0])
        height = abs(points[1][1]) + abs(points[3][1])
        plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
        plane.paint_uniform_color(color)
        plane.translate([points[1][0], points[1][1], scale])
        plane.transform(T)
        output.append(plane)

    return output


class Visualization:
    def __init__(self):
        self._vis = None

    def add_points(self, xyz, rgb, remove_statistical_outlier=True):
        ''' xyz: shape (N, 3), rgb: shape (N, 3) and [0-1] '''
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # remove obvious outliers
        if remove_statistical_outlier:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0,
            )

        self._vis.add_geometry(pcd)
        self._vis.poll_events()
        self._vis.update_renderer()

    def add_camera(self, K, R, t, width, height, scale=1.0, color=[0.8, 0.2, 0.8]):
        camera_mesh = draw_camera(K, R, t, width, height, scale, color)
        for x in camera_mesh:
            self._vis.add_geometry(x)
        self._vis.poll_events()
        self._vis.update_renderer()

    def add_mesh(self, x):
        self._vis.add_geometry(x)
        self._vis.poll_events()
        self._vis.update_renderer()

    def add_origin_axis(self):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self._vis.add_geometry(axis)
        self._vis.poll_events()
        self._vis.update_renderer()

    def create_window(self, name=""):
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(name)

    def show(self):
        self._vis.poll_events()
        self._vis.update_renderer()
        self._vis.run()
        self._vis.destroy_window()
