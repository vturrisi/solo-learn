python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --encoding_dim 2048 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --asymmetric_augmentations \
    --scale_loss 0.1 \
    --name barlow \
    --method barlow_twins \
    --dali \
    --project contrastive_learning
