python3 ../main_contrastive.py \
    imagenet \
    resnet50 \
    --data_folder /data/datasets \
    --train_dir imagenet/train \
    --val_dir imagenet/val \
    --epochs 100 \
    --optimizer sgd \
    --scheduler cosine \
    --lr 0.05 \
    --classifier_lr 0.05 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --encoding_dim 2048 \
    --zero_init_residual \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --name simsiam-resnet50-imagenet-same-parameters \
    --method simsiam \
    --no_lr_scheduler_for_pred_head \
    --dali \
    --project contrastive_learning

# python3 ../main_linear.py \
#     imagenet \
#     resnet50 \
#     --data_folder /data/datasets \
#     --train_dir imagenet/train \
#     --val_dir imagenet/val \
#     --epochs 90 \
#     --optimizer sgd \
#     --lars \
#     --scheduler cosine \
#     --lr 0.02 \
#     --weight_decay 0 \
#     --batch_size 2048 \
#     --gpus 0 1 \
#     --num_workers 10 \
#     --name simsiam-resnet50-imagenet-same-parameters-linear-eval \
#     --pretrained_feature_extractor trained_models/3hgyrb2v \
#     --dali \
#     --project contrastive_learning
