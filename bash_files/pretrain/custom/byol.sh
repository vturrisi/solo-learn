# Train without labels.
# To train with labels, simply remove --no_labels
# --val_dir is optional and will expect a directory with subfolder (classes)
# --dali flag is also supported
python3 ../../../main_pretrain.py \
    --dataset custom \
    --encoder resnet18 \
    --data_dir PATH_TO_DIR \
    --train_dir PATH_TO_TRAIN_DIR \
    --no_labels \
    --max_epochs 400 \
    --gpus 0,1 \
    --accelerator ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --name byol-400ep-custom \
    --entity unitn-mhug \
    --project solo-learn \
    --wandb \
    --save_checkpoint \
    --method byol \
    --output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 8192 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0
