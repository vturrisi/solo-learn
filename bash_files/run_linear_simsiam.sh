python3 ../main_linear.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 90 \
    --optimizer sgd \
    --scheduler cosine \
    --lr 0.02 \
    --weight_decay 0 \
    --batch_size 2048 \
    --gpus 0 1 \
    --lars \
    --num_workers 10 \
    --name simsiam-linear-eval \
    --pretrained_feature_extractor trained_models/fsjdzl68 \
    --project contrastive_learning
