from typing import List, Tuple, Dict

import argparse
import json

import numpy as np
import open3d as o3d
from scannetpp_tools.common.scene_release import ScannetppScene_Release
from scannetpp_tools.common.utils.colmap import read_points3D_text
from utils.visualization import Visualization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--load_mesh", action="store_true")
    parser.add_argument("--load_camera", action="store_true")
    parser.add_argument("--camera_subsample", type=int, default=2)
    parser.add_argument("--load_colmap_points", action="store_true")
    return parser.parse_args()


def parse_frames(frames: List[Dict]) -> Tuple[List[str], List[np.ndarray]]:
    file_names = []
    camera_to_worlds = []
    for frame in frames:
        # Convert the poses from nerfstudio to colmap convention
        image_name = frame["file_path"]
        camera_to_world = np.array(frame["transform_matrix"])
        camera_to_world[2, :] *= -1
        camera_to_world = camera_to_world[np.array([1, 0, 2, 3]), :]
        camera_to_world[0:3, 1:3] *= -1

        file_names.append(image_name)
        camera_to_worlds.append(camera_to_world)
    return file_names, camera_to_worlds


def main(args):
    vis = Visualization()
    vis.create_window("ScanNet++ Scene Viewer")
    scene = ScannetppScene_Release(data_root=args.data_root, scene_id=args.scene_id)

    if args.load_camera:
        json_path = scene.dslr_nerfstudio_transform_undistorted_path
        with open(json_path, "r") as f:
            transforms = json.load(f)
        height = int(transforms["h"])
        width = int(transforms["w"])
        fx = float(transforms["fl_x"])
        fy = float(transforms["fl_y"])
        cx = float(transforms["cx"])
        cy = float(transforms["cy"])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        train_frames = transforms["frames"]
        test_frames = transforms["test_frames"]

        train_file_names, train_camera_to_worlds = parse_frames(train_frames)
        test_file_names, test_camera_to_worlds = parse_frames(test_frames)

        for camera_to_world in train_camera_to_worlds[::args.camera_subsample]:
            vis.add_camera(
                K,
                camera_to_world[:3, :3],
                camera_to_world[:3, 3],
                width,
                height,
                scale=0.1,
                color=[0.8, 0.2, 0.8],
            )
        for camera_to_world in test_camera_to_worlds:
            vis.add_camera(
                K,
                camera_to_world[:3, :3],
                camera_to_world[:3, 3],
                width,
                height,
                scale=0.1,
                color=[0.2, 0.8, 0.2],
            )

    if args.load_mesh:
        mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path))
        vis.add_mesh(mesh)

    if args.load_colmap_points:
        points3D = read_points3D_text(scene.dslr_colmap_dir / "points3D.txt")
        xyz = []
        rgb = []
        for point in points3D.values():
            xyz.append(point.xyz)
            rgb.append(point.rgb)
        xyz = np.array(xyz)
        rgb = np.array(rgb) / 255.0
        vis.add_points(xyz, rgb)

    vis.add_origin_axis()
    vis.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
