export CUDA_VISIBLE_DEVICES=1

SCENES="7b4cb756d4 1f7cbbdde1"

python train.py \
    --data_root /workspace/data/scannetpp/dev \
    --output_root /workspace/project/scannetpp/output \
    --test_every 10000 \
    --scene_ids $SCENES \