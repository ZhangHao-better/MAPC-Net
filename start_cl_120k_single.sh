#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
/home/zhlinux/anaconda3/envs/rtm_linux/bin/python tools/train.py \
    configs/ascformer/ascformer_rtm_cl_120k.py \
    2>&1 | tee train_cl_120k_single.log
