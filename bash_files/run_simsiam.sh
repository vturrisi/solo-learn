python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /data/datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 30.0 \
    --weight_decay 1e-4 \
    --batch_size 2 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_mlp 2048 \
    --pred_hidden_mlp 512 \
    --encoding_size 2048 \
    --no_projection_bn \
    --zero_init_residual \
    --name simsiam \
    --method simsiam \
    --no_lr_scheduler_for_pred_head \
    --dali \
    --project debug
