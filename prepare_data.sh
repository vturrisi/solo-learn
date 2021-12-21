#!/bin/bash

####
#
# Usage: ./prepare_data.sh VAL for using a validation set from the training set. Otherwise use test set.
#
###

MODE=$1

[[ $HOSTNAME == login* ]] || [[ $SLURM_CLUSTER_NAME == mila ]]; mila=$?;

if [ $mila == 0 ];
then
  echo "In Mila - $mila"
  SRC_TRAIN=/network/datasets/imagenet/ILSVRC2012_img_train.tar
  SRC_VAL=/network/scratch/l/lavoiems/data/imagenet_val.tar
else
  echo "In CC - $mila"
  SRC_TRAIN=/network/scratch/lavoiems/data/ILSVRC2012_img_train.tar
  SRC_VAL=/network/scratch/lavoiems/data/imagenet_val.tar
fi
TRG=$SLURM_TMPDIR/data
TRG_TRAIN=$TRG/train
TRG_VAL=$TRG/val
TRG_TMP=$SLURM_TMPDIR/tmp_data
mkdir -p $TRG
mkdir -p $TRG_TRAIN
mkdir -p $TRG_TMP

#echo "Extracting IMAGENET from $SRC_TRAIN into $TRG_TMP"
#tar -xf $SRC_TRAIN -C $TRG_TMP
#
#echo "Extracting training set from $TRG_TMP into $TRG_TRAIN"
#for D in `ls $TRG_TMP`; do
#  mkdir -p "$TRG_TRAIN/${D%.*}"
#  tar -xf $TRG_TMP/$D -C "$TRG_TRAIN/${D%.*}" &
#done
#wait

if [[ $MODE == "VAL" ]];
then
  echo "Taking 10 samples/class as validation sample"
  mkdir -p $TRG_VAL
  for D in `ls $TRG_TRAIN`; do
    mkdir -p $TRG_VAL/$D
    ls $TRG_TRAIN/$D/ | head -10 | xargs -i mv $TRG_TRAIN/$D/{} $TRG_VAL/$D/ &
  done
  wait
else
  echo "Extracting test set from $SRC_VAL into $TRG"
  tar -xf $SRC_VAL -C $TRG
fi

