python3 ../../../main_contrastive.py \
    --dataset $1 \
    --encoder resnet18 \
<<<<<<< HEAD
    --data_dir ../datasets \
=======
    --data_folder datasets \
>>>>>>> 3968fd35fd34a3d4a976130b42a6d933de823f32
    --max_epochs 1000 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --num_workers 3 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --name mocov2plus-$1 \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --method mocov2plus \
    --proj_hidden_dim 2048 \
    --queue_size 32768 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier