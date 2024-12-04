#!/bin/bash

log_dir=../../log/grasp_v6
mkdir -p $log_dir
export RANK_SIZE=$1
export rank_start=$2
export GLOG_v=3
for ((i=0; i<$RANK_SIZE; i++))
do
        j=$((rank_start+i))
        export DEVICE_ID=$j
        export RANK_ID=$i
        /share/archiconda3/envs/ms2.2/bin/python infer_xl_ccpdb.py > ${log_dir}/log_high_${i}.log 2>&1 &
done