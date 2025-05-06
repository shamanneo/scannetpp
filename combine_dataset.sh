SCENES="9071e139d9 927aacd5d1 7b4cb756d4 1f7cbbdde1 0a7cc12c0e"

python combine_dataset.py \
    --data_dir /workspace/data/scannetpp/dev \
    --scene_ids $SCENES \
    --data_name comb \
    --dslr_img resized_undistorted_images \
    --iphone_img rgb
