#!/usr/bin/env sh

DATA_SET=image
# define data source for train and test
DATA_FILE=cifar100_fine_R10

DEBUG=false

TRAIN_LIST=/scratch/dutta/cifarR20_trainval_train.txt
VAL_LIST=/scratch/dutta/test_fine_labels.txt

TRAIN_BASE_DIR=/scratch/dutta/cifar\-100\-train
VAL_BASE_DIR=/scratch/dutta/cifar\-100\-test

WIDTH=96
WIDTH_MULT=2

CUDA_VISIBLE_DEVICES=0
th main.lua \
  -dataset $DATA_SET \
  -dataFile $DATA_FILE \
  -trainList $TRAIN_LIST \
  -trainBaseDir $TRAIN_BASE_DIR \
  -valList $VAL_LIST \
  -valBaseDir $VAL_BASE_DIR \
  -width $WIDTH \
  -widthMult $WIDTH_MULT \
  -debug $DEBUG
