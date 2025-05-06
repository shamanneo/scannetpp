import json
import shutil
import shutil
import argparse

from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--scene_ids", type=str, nargs="+", required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--dslr_img", type=str, required=True)
    parser.add_argument("--iphone_img", type=str, required=True)
    
    args = parser.parse_args()
    return args

def copy_files(src_dir, dst_dir):
    count = 0
    for file in src_dir.glob("*"):
        if file.is_file():
            dst = dst_dir / file.name
            shutil.copy2(file, dst)
            count += 1
    return count

if __name__ == "__main__":
    args = get_args()
    for scene_id in args.scene_ids:
        print(f"[INFO] Processing {scene_id}")
        SCENE_DIR = Path(args.data_dir) / scene_id

        output_dir = SCENE_DIR / args.data_name
        output_dir.mkdir(parents = True, exist_ok = True)
        dslr_dir = SCENE_DIR / "dslr"
        iphone_dir = SCENE_DIR / "iphone"

        # combine images
        combined_img_dir = output_dir / "images"
        combined_img_dir.mkdir(parents = True, exist_ok = True)
        dslr_count = copy_files(dslr_dir / args.dslr_img, combined_img_dir)
        iphone_img_dir = iphone_dir / args.iphone_img
        if not iphone_img_dir.exists():
            raise FileNotFoundError(f"The directory {iphone_img_dir} does not exist.")
        iphone_count = copy_files(iphone_img_dir, combined_img_dir)
        print(f"[INFO] Saved {dslr_count} DSLR images")
        print(f"[INFO] Saved {iphone_count} iPhone images")

        # combine transforms.json
        dslr_json_file = "nerfstudio/transforms_undistorted.json"
        iphone_json_file = "nerfstudio/transforms.json"
        with open(dslr_dir / dslr_json_file, 'r') as dslr_json, open(iphone_dir / iphone_json_file, 'r') as iphone_json: 
            dslr_transform = json.load(dslr_json)
            iphone_transform = json.load(iphone_json)
            # intrinsic params
            output_transform = {
                "dslr": {
                    "fl_x": dslr_transform["fl_x"],
                    "fl_y": dslr_transform["fl_y"],
                    "cx": dslr_transform["cx"],
                    "cy": dslr_transform["cy"],
                    "w": dslr_transform["w"],
                    "h": dslr_transform["h"],
                },
                "iphone": {
                    "fl_x": iphone_transform["fl_x"],
                    "fl_y": iphone_transform["fl_y"],
                    "cx": iphone_transform["cx"],
                    "cy": iphone_transform["cy"],
                    "w": iphone_transform["w"],
                    "h": iphone_transform["h"],
                },
                "camera_model": dslr_transform["camera_model"], # pinhole
                "frames": [],
                "test_frames": []
            }

            for frame in dslr_transform["frames"]:
                frame["ctype"] = "dslr"
            for frame in dslr_transform["test_frames"]:
                frame["ctype"] = "dslr" 
            for frame in iphone_transform["frames"]:
                frame["ctype"] = "iphone" 

            output_transform["frames"].extend(dslr_transform["frames"])
            output_transform["frames"].extend(iphone_transform["frames"])
            # Using dslr test frames for evaluation
            output_transform["test_frames"].extend(dslr_transform["test_frames"])

            output_transform_dir = output_dir / "nerfstudio/transforms.json"
            output_transform_dir.parent.mkdir(parents=True, exist_ok=True)

            with open(output_transform_dir, 'w') as f:
                json.dump(output_transform, f, indent=4)