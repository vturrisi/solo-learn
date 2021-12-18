export DETECTRON2_DATASETS=/data/datasets

# good results for BYOL
python3 train_object_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
    --num-gpus 2 MODEL.WEIGHTS ./detectron_model.pkl SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.1
