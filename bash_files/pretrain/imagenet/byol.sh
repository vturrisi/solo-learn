python3 ../../../main_pretrain.py \
    --dataset imagenet \
    --encoder resnet50 \
    --data_dir /datasets \
    --train_dir ILSVRC2012/train \
    --val_dir ILSVRC2012/val \
    --max_epochs 100 \
    --gpus 0,1 \
    --accelerator ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.45 \
    --accumulate_grad_batches 16 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 8 \
    --dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --name byol-resnet50-imagenet-100epochs \
    --entity unitn-mhug \
    --project solo-learn \
    --wandb \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier
