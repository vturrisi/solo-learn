python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /data/datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 800 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --lars \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_mlp 2048 \
    --pred_hidden_mlp 512 \
    --encoding_size 2048 \
    --no_projection_bn \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --temperature 0.2 \
    --zero_init_residual \
    --name simsiam \
    --method simsiam \
    --dali \
    --project debug

# --no_lr_scheduler_for_pred_head \
