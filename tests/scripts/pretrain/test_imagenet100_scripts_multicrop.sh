TRAIN_PATH=$1
VAL_PATH=$2
FORMAT=$3
METHODS=("byol" "simclr" "supcon")

for method in ${METHODS[@]}; do
    echo Running $method
    python3 main_pretrain.py \
        --config-path scripts/pretrain/imagenet-100-multicrop/ \
        --config-name $method \
        max_epochs=2 \
        devices=[0] \
        optimizer.batch_size=32 \
        data.train_path=$TRAIN_PATH \
        data.val_path=$VAL_PATH \
        data.format=$FORMAT \
        ++auto_resume.enabled=False \
        ++wandb.enabled=False \
        ++limit_train_batches=30 \
        ++limit_val_batches=30 \
        ++method_kwargs.warmup_teacher_temperature_epochs=0
    echo ---------------------------------
done
