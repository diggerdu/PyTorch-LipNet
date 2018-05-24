#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2
expName=new
selfPath=`realpath $0`
export expPath=`realpath .`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python testDataLoader.py \
 --dumpPath /data1/dumpFile/prominate \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 24 \
 --batchSize  96 \
 --niter 10000 --niter_decay 30 \
 --lr 0.001\
 --gpu_ids 0,1,2 
# --continue_train --which_epoch 86
#  --serial_batches
