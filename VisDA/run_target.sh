#!/bin/bash

T=`date +%m%d%H%M`
# ROOT=../../
# export PODROOT=$ROOT
#
# export PYTHONPATH=$ROOT:$PYTHONPATH

python target.py --gpu_id 0,1,2,3 --output ./experiments/VISDA-C/target/ --output_src ./experiments/VISDA-C/source/ --da uda --dset VISDA-C --net resnet101 --lr 5e-4 --s 0 --cls_par 0.6 --lamda_m 1 --p_threshold 0.97 --T $T --ent_par 0.1 --lamda_ad 0.3 --ad_method EMMD 