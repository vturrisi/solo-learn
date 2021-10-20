export DETECTRON2_DATASETS=/data/datasets

python3 train_object_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
    --num-gpus 2 MODEL.WEIGHTS ./detectron_model.pkl
