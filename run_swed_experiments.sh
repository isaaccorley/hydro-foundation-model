#!/bin/bash

python train_pct_runs.py --config configs/swed/hydro_msi.yaml --device 0 --max_epochs 1 --output swed_hydro_msi_1.csv &
python train_pct_runs.py --config configs/swed/unet_resnet50_msi.yaml --device 1 --max_epochs 1 --output swed_unet_resnet50_msi_1.csv &
python train_pct_runs.py --config configs/swed/unet_swinv2_imagenet_msi.yaml --device 2 --max_epochs 1 --output swed_unet_swinv2_imagenet_msi_1.csv &
python train_pct_runs.py --config configs/swed/unet_swinv2_satlas_msi.yaml --device 3 --max_epochs 1 --output swed_unet_swinv2_satlas_msi_1.csv &

wait
echo "All training jobs completed."