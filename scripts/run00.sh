export CUDA_VISIBLE_DEVICES=0

SCENES="9071e139d9 927aacd5d1"

python train.py \
    --data_root /workspace/data/scannetpp/dev \
    --output_root /workspace/project/scannetpp/output \
    --test_every 10000 \
    --scene_ids $SCENES \