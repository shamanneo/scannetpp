# 3D Gaussian Splatting on ScanNet++ dataset
This repository contains the code for training and evaluating 3D Gaussian Splatting on the ScanNet++ dataset. The code is based on the original 3D Gaussian Splatting [github](https://github.com/graphdeco-inria/gaussian-splatting) and adapted to work with the ScanNet++ dataset.

## Requirements
The code is tested with Python 3.9 and requires the following packages:
```
conda create -n 3dgs-demo python=3.9
conda activate 3dgs-demo

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install opencv-python plyfile tqdm open3d
```


## Dataset Preparation
Apply and download the ScanNet++ dataset from [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/). The dataset after download should be organized as follows:
```
scannet_download
/cluster/andram/yliu/scannetpp_v2_official
├── data
│   ├── SCENE_ID1
│   ├── SCENE_ID2
│   ├── SCENE_ID3
├── metadata
└── splits
```

**NOTE**: The code is based on undistorted DSLR images with the following format which is automatically provided by downloading the dataset after **2025.04.29** using the new download script. Otherwise, you can use the provided script `scannetpp_tools/dslr/undistort.py` to undistort the images following the [instruction](https://github.com/scannetpp/scannetpp?tab=readme-ov-file#undistortion-convert-fisheye-images-to-pinhole-with-opencv).
```
scene_id/dslr
├── colmap
├── nerfstudio
│   └── transforms_undistorted.json
├── resized_undistorted_images
└── resized_undistorted_masks
```

## Visualization
To visualize the camera poses and the meshes, you can use the provided viewer script. Make sure to have Open3D installed.

```sh
python viewer.py \
    --data_root [SCANNET++ DATA ROOT] \
    --scene_id 39f36da05b \
    --load_camera \
    --load_mesh
```

This will load the camera poses and the mesh for the specified scene ID and visualize them using Open3D as the following example:
![visualization](assets/viewer_example1.png)


## Running
To run 3DGS on a scene in ScanNet++ dataset, you can use the following command:

```sh
python train.py \
    --data_root [SCANNET++ DATA ROOT] \
    --output_path [OUTPUT DATA ROOT] \
    --scene_id 39f36da05b
```

At the end of the training, it would render the testing images and stores them in `[OUTPUT DATA ROOT]/submission` folder, following the [official submission format](https://kaldir.vc.in.tum.de/scannetpp/benchmark/docs). Once you have all the testing scenes, this folder can be then zipped and submitted to the ScanNet++ NVS benchmark server.


## Evaluation
To evaluate the results (on validation sets), you can use the provided evaluation script in `scannetpp_tools` ([ScanNet++ Toolbox](https://github.com/scannetpp/scannetpp?tab=readme-ov-file#novel-view-synthesis-evaluation-dslr)).
```sh
cd scannetpp_tools
# Evaluate on a single scene
python -m eval.nvs \
    --data_root /cluster/andram/yliu/scannetpp_v2_official/data \
    --scene_id [SCENE_ID] \
    --pred_dir [OUTPUT DATA ROOT]/submission

# Evaluate on the whole validation set
python -m eval.nvs \
    --data_root [SCANNET++ DATA ROOT] \
    --split [SCANNET++ SPLIT]/nvs_sem_val.txt \
    --pred_dir [OUTPUT DATA ROOT]/submission
```