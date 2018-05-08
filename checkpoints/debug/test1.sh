#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
expName=debug
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python true.py \
 --dumpPath /data1/dumpFile/prominate \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 6 \
 --batchSize  12\
 --niter 10000 --niter_decay 30 \
 --lr 0.0000001\
 --gpu_ids 0,1\
 --continue_train \
 --which_epoch $1
#  --serial_batches
#--input_nc 1 --output_nc 1 \
