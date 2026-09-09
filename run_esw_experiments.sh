#!/bin/bash

# python train_pct_runs.py --config configs/earthsurfacewater/hydro_msi.yaml --device 0 --train_fractions 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256 0.0512 0.1012 --max_epochs 10 --output esw_hydro_msi_10.csv &
# python train_pct_runs.py --config configs/earthsurfacewater/unet_resnet50_msi.yaml --device 1 --train_fractions 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256 0.0512 0.1012 --max_epochs 10 --output esw_unet_resnet50_msi_10.csv &
# python train_pct_runs.py --config configs/earthsurfacewater/unet_swinv2_imagenet_msi.yaml --device 2 --train_fractions 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256 0.0512 0.1012 --max_epochs 10 --output esw_unet_swinv2_imagenet_msi_10.csv &
# python train_pct_runs.py --config configs/earthsurfacewater/unet_swinv2_satlas_msi.yaml --device 3 --train_fractions 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256 0.0512 0.1012 --max_epochs 10 --output esw_unet_swinv2_satlas_msi_10.csv &

python train_pct_runs.py --config configs/earthsurfacewater/hydro_rgb.yaml --device 0 --train_fractions 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256 0.0512 0.1012 --max_epochs 1 --output results/esw_hydro_rgb_1.csv &
python train_pct_runs.py --config configs/earthsurfacewater/unet_resnet50_rgb.yaml --device 1 --train_fractions 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256 0.0512 0.1012 --max_epochs 1 --output results/esw_unet_resnet50_rgb_1.csv &
python train_pct_runs.py --config configs/earthsurfacewater/unet_swinv2_imagenet_rgb.yaml --device 2 --train_fractions 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256 0.0512 0.1012 --max_epochs 1 --output results/esw_unet_swinv2_imagenet_rgb_1.csv &
python train_pct_runs.py --config configs/earthsurfacewater/unet_swinv2_satlas_rgb.yaml --device 3 --train_fractions 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256 0.0512 0.1012 --max_epochs 1 --output results/esw_unet_swinv2_satlas_rgb_1.csv &



wait
echo "All training jobs completed."