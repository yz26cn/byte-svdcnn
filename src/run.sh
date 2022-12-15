#!/usr/bin/env bash
cd ../
batch_size=64
maxlen=1024
epochs=50
lr=0.01
lr_halve_interval=3
gamma=0.9
snapshot_interval=3
gpuid=0
nthreads=4
 
dataset="noah"
data_folder="datasets/${dataset}/svdcnn_0.05"
depth=17
model_folder="models/${dataset}_${depth}_0.05"
shortcut=True

python -m src.main --dataset ${dataset} \
                         --model_folder ${model_folder} \
                         --data_folder ${data_folder} \
                         --depth ${depth} \
                         --maxlen ${maxlen} \
                         --batch_size ${batch_size} \
                         --epochs ${epochs} \
                         --lr ${lr} \
                         --lr_halve_interval ${lr_halve_interval} \
                         --snapshot_interval ${snapshot_interval} \
                         --gamma ${gamma} \
                         --gpuid ${gpuid} \
                         --nthreads ${nthreads} \
                         --shortcut ${shortcut} \