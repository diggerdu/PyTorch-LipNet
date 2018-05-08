#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
expName=new
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python testDataLoader.py \
 --dumpPath /data1/dumpFile/prominate \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 56 \
 --batchSize  108 \
 --niter 10000 --niter_decay 30 \
 --lr 0.01\
 --gpu_ids 0,1,2,3
# --continue_train
#  --serial_batches
