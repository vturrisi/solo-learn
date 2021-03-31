python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /home/vturrisi/Documents/hmdb_ucf/hmdb/test \
    --train_dir fencing \
    --val_dir fencing \
    --epochs 100 \
    --optimizer sgd \
    --scheduler cosine \
    --lr 0.05 \
    --classifier_lr 30.0 \
    --weight_decay 1e-4 \
    --batch_size 10 \
    --gpus 0 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --encoding_dim 2048 \
    --zero_init_residual \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --jit_transforms \
    --name debug \
    --method simsiam \
    --no_lr_scheduler_for_pred_head \
    --project debug
