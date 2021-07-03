python3 ../../../main_contrastive.py \
    --dataset $1 \
    --encoder resnet18 \
    --data_dir /data/datasets \
    --max_epochs 1000 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 5 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --asymmetric_augmentations \
    --name nnclr-$1 \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --method nnclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --pred_hidden_dim 4096 \
    --output_dim 256 \
    --queue_size 65536