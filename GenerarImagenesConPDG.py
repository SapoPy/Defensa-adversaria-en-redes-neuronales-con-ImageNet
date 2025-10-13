import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.models import resnet34, ResNet34_Weights

from ImageNet100ValDataset import *

# ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def pgd_attack(model, images, labels, eps=0.05, alpha=2/255, iters=10,
               random_start=True, targeted=False, device=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    images: tensor (B,C,H,W) NORMALIZADO (modelo espera Normalize already).
    Devuelve x_adv (normalizado) con misma forma que images.
    eps, alpha en pixel-space (ej. 8/255).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()
    images = images.to(device)
    labels = labels.to(device)

    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, -1, 1, 1)

    # eps/alpha en espacio normalizado (dividir por std por canal)
    eps_norm = (eps / std_t).to(device)
    alpha_norm = (alpha / std_t).to(device)

    min_norm = ((0.0 - mean_t) / std_t)
    max_norm = ((1.0 - mean_t) / std_t)

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()
    if random_start:
        rand_pert = torch.empty_like(x_adv).uniform_(-eps, eps).to(device)   # pixel-space uniform
        rand_pert_norm = rand_pert / std_t
        x_adv = torch.clamp(x_adv + rand_pert_norm, min_norm, max_norm).detach()

    x_adv.requires_grad_(True)

    for _ in range(iters):
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, labels)
        if targeted:
            loss = -loss

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()
        grad = x_adv.grad.data

        if targeted:
            x_adv = x_adv - alpha_norm * grad.sign()
        else:
            x_adv = x_adv + alpha_norm * grad.sign()

        # proyectar L_inf en espacio normalizado alrededor de x_orig
        x_adv = torch.max(torch.min(x_adv, x_orig + eps_norm), x_orig - eps_norm)
        # clamp a rango válido
        x_adv = torch.max(torch.min(x_adv, max_norm), min_norm).detach()
        x_adv.requires_grad_(True)

    return x_adv.detach()


def generar_dataset_pgd_guardado_como_dataset(model, dataset, out_dir,
                                               eps=8/255, alpha=2/255, iters=10,
                                               batch_size=32, device=None, overwrite=False,
                                               mean=IMAGENET_MEAN, std=IMAGENET_STD,
                                               targeted=False, target_idx=None):
    """
    Recorre `dataset`, genera adversarios (no dirigido por defecto)
    y guarda imágenes denormalizadas en out_dir/wnid/*.png.
    Devuelve (out_dir_str, ImageNet100ValDataset(out_dir, transform=dataset.transform))
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, -1, 1, 1)

    total_imgs = len(dataset)
    idx_global = 0

    print(f"\nGenerando imágenes adversarias PGD ({total_imgs} en total):\n")

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        if targeted:
            if target_idx is None:
                raise ValueError("target_idx required for targeted=True")
            target_labels = torch.full((imgs.size(0),), target_idx, device=device, dtype=torch.long)
            imgs_adv = pgd_attack(model, imgs, target_labels, eps=eps, alpha=alpha, iters=iters,
                                  random_start=True, targeted=True, device=device, mean=mean, std=std)
        else:
            imgs_adv = pgd_attack(model, imgs, labels, eps=eps, alpha=alpha, iters=iters,
                                  random_start=True, targeted=False, device=device, mean=mean, std=std)

        imgs_adv_pixel = (imgs_adv * std_t + mean_t).clamp(0.0, 1.0).cpu()

        for b in range(imgs_adv_pixel.size(0)):
            try:
                orig_path = dataset.samples[idx_global + b][0]
                wnid = Path(orig_path).parent.name
                fname = Path(orig_path).name
            except Exception:
                wnid = str(labels[b].item())
                fname = f"adv_{idx_global + b:06d}.png"

            class_dir = out_dir / wnid
            class_dir.mkdir(parents=True, exist_ok=True)
            save_path = class_dir / f"{Path(fname).stem}_pgd.png"

            if overwrite or not save_path.exists():
                save_image(imgs_adv_pixel[b], save_path)

        idx_global += imgs.size(0)

        # --- NUEVO: imprimir progreso ---
        print(f"Procesadas: {min(idx_global, total_imgs)}/{total_imgs} imágenes...", end="\r")

    print(f"\n Generación completada: {idx_global} imágenes adversarias guardadas en '{out_dir}'\n")

    labels_json = getattr(dataset, "labels_json", "Labels.json")
    adv_ds = ImageNet100ValDataset(str(out_dir), transform=dataset.transform, labels_json=labels_json)
    return str(out_dir), adv_ds



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device).eval()

    VAL_DIR = "val.X"   # carpeta con subfolders wnid/
    ds = ImageNet100ValDataset(VAL_DIR, transform=transform)

    # parámetros PGD (no dirigido)
    out_dir, adv_ds = generar_dataset_pgd_guardado_como_dataset(
        model, ds, out_dir="val_pgd_untargeted_015",
        eps=0.15, alpha=2/255, iters=10,
        batch_size=32, device=device, overwrite=False,
    )

