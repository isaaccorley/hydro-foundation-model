# Hydro -- A Foundation Model for Water in Satellite Imagery

<p align="center">
    <img src="./assets/image.jpg" width="500"/><br/>
</p>

This repo started out as mostly /s and is a work in progress side project, but maybe something useful or interesting will happen.

### Motivation

There are many foundation models for remote sensing but nearly all of them focus on imagery of land. The earth is made of ~71% water. **ML 4 Water needs some more love and attention**. Therefore, we perform pretraining on a large-scale dataset of Sentinel-2 imagery containing water bodies for use in downstream applications such as bathmetry and hydrology.

### Progress

<p align="center">
    <img src="./assets/simmim.jpg" width="400"/><br/>
</p>

We pretrain a [Swin v2 Transformer](https://arxiv.org/abs/2111.09883) encoder using the **SimMIM** method from the paper, ["SimMIM: A Simple Framework for Masked Image Modeling"](https://arxiv.org/abs/2111.09886), which is an efficient variation of the [Masked Autoencoder (MAE)](https://arxiv.org/abs/2111.06377) self-supervised learning framework.

Our pretraining dataset consists of 100k sampled 256x256 Sentinel-2 patches containing water from around the globe.

### Usage

Clone the repo with submodules

```
git clone --recurse-submodules https://github.com/isaaccorley/hydro-foundation-model.git
```

This repo really only requires packages that torchgeo installs so go ahead and `pip install torchgeo`

To pretrain a model you can run the following. Note to change some parameters you can edit the config file in the `configs/hydro/` folder.

```bash
cd Swin-Transformer
bash hydro_pretrain.sh
```

To load the model and generate image embeddings see `embed.ipynb`. For now only image level embeddings are supported but intermediate layer embeddings will be added soon.

### Evaluation

We plan to evaluate the model on a few bathymetry, hydrology, and other benchmark datasets. This repo contains dataset code for evaluating on the [Marine Debris Archive (MARIDA)](https://marine-debris.github.io/) dataset which is in progress.

### Cite

If you use our pretrained models in you work please cite the following:

```bash
@misc{Corley:2024,
  Author = {Isaac Corley, Caleb Robinson},
  Title = {Hydro Foundation Model},
  Year = {2024},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/isaaccorley/hydro-foundation-model}}
}
```
