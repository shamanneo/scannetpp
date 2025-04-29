#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from dataset import ScannetppDataset
from utils.graphics_utils import BasicPointCloud


class Scene:
    def __init__(
        self,
        args: ModelParams,
        dataset: ScannetppDataset,
        gaussians: GaussianModel,
    ):
        # self.model_path = args.model_path
        self.gaussians = gaussians
        self.loaded_iter = None

        # self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras_extent = dataset.get_nerfpp_norm()["radius"]
        xyz, rgb = dataset.get_point_cloud()
        pcd = BasicPointCloud(
            points=xyz,
            colors=rgb,
            normals=None,
        )
        self.gaussians.create_from_pcd(
            pcd,
            dataset.get_image_names(),
            self.cameras_extent,
        )

    def save(self, iteration, output_path):
        point_cloud_path = os.path.join(output_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(output_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)
