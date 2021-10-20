export DETECTRON2_DATASETS=/data/datasets

python3 train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
    --num-gpus 8 MODEL.WEIGHTS ./detectron_model.pkl
