TRAIN_PATH=$1
VAL_PATH=$2
FORMAT=$3
METHODS=("mae")

# first run ../pretrain/test_imagenet_scripts.sh and then fill the paths here
# escape path with \"PATH-HERE\"
# https://hydra.cc/docs/advanced/override_grammar/basic/#:~:text=Hydra%20supports%20both%20double%20quotes,quoted%20string%2C%20use%20%5C%22%20)
PRETRAINED_PATHS=(
    \"PATH-TO-MAE-MODEL\"
)

for i in ${!METHODS[@]}; do
    method=${METHODS[i]}
    pretrained_path=${PRETRAINED_PATHS[i]}
    echo Running $method
    python3 main_linear.py \
        --config-path scripts/finetune/imagenet-100/ \
        --config-name $method \
        pretrained_feature_extractor=$pretrained_path \
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
