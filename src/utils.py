from tqdm import tqdm
import torch
import numpy as np


@torch.no_grad()
@torch.inference_mode()
def extract_features(model, dataloader, device, transforms=None):
    x_all, y_all = [], []

    for batch in tqdm(dataloader, total=len(dataloader)):
        images = batch["image"].to(device)
        labels = batch["label"].detach().cpu().numpy()

        if transforms is not None:
            images = transforms(images)
            features = model.forward_features(images)

            if features.ndim > 2:
                features = features.mean(dim=(1, 2))

            features = features.detach().cpu().numpy()

        x_all.append(features)
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all
