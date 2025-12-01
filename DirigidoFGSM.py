# FGSM_targeted.py
import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnet34, ResNet34_Weights
from ImageNet100ValDataset import ImageNet100ValDataset
from utils import *

# ---------------------------
# Constantes ImageNet
# ---------------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ---------------------------
# mapear WNID -> índice del modelo (0..999)
# ---------------------------
def wnid_to_model_index(wnid, weights, labels_json="Labels.json"):
    with open(labels_json, "r") as f:
        wnid2name = json.load(f)
    if wnid not in wnid2name:
        raise ValueError(f"WNID {wnid} no está en {labels_json}")
    target_name = wnid2name[wnid].split(",")[0].strip().lower()
    imagenet_classes = weights.meta["categories"]
    # primera búsqueda exacta (case-insensitive)
    for i, cname in enumerate(imagenet_classes):
        if cname.lower().strip() == target_name:
            return i
    # búsqueda parcial por token
    token = target_name.split()[0]
    for i, cname in enumerate(imagenet_classes):
        if token in cname.lower():
            return i
    raise ValueError(f"No pude mapear {wnid} ('{target_name}') -> índice en weights.meta['categories'].")

# ---------------------------
# FGSM dirigido (usa grad calculado sobre entrada normalizada)
# ---------------------------
def generar_dataset_fgsm_dirigido(model, dataset, eps, target_idx, out_dir,
                                  device=None, print_every=50, save_examples_n=3):
    """
    Genera imágenes adversarias FGSM dirigidas a `target_idx` (índice en 0..999 del modelo).
    Guarda imágenes denormalizadas (pixel-space) en out_dir/wnid/*.png y devuelve ImageNet100ValDataset apuntando a out_dir.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    shown = 0

    std_t = torch.tensor(STD, device=device).view(1, -1, 1, 1)   # para convertir grad normalized -> pixel-space
    for i in range(total):
        # dataset devuelve img_norm (ya NORMALIZADA si dataset.transform lo hace), y label_local
        img_norm, label_local = dataset[i]  # img_norm: (C,H,W) normalized (tensor)
        # asegurar tensors en CPU/device según cálculo
        # quedamos con img_norm en CPU, pero para forward necesitamos en device
        x = img_norm.unsqueeze(0).to(device).detach()
        x.requires_grad_(True)

        # targeted FGSM: usamos la etiqueta objetivo (target_idx) en el loss
        target = torch.tensor([target_idx], device=device, dtype=torch.long)

        out = model(x)
        loss = F.cross_entropy(out, target)   # minimizar loss hacia target
        # IMPORTANTE: en ataque dirigido invertimos la suma: restamos grad.sign() en pixel-space
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        # grad en espacio normalizado (dL/dx_norm)
        grad_norm = x.grad.detach().cpu().squeeze(0)  # (C,H,W) on CPU

        # convertir grad a pixel-space: dL/dx_pixel = dL/dx_norm * (1/std)
        std_cpu = torch.tensor(STD).view(1, -1, 1, 1)
        grad_pixel = grad_norm / std_cpu.squeeze(0)  # (C,H,W) CPU

        # denormalizar imagen original a pixel-space para sumar la perturbación
        img_pixel = denormalize_tensor(img_norm.unsqueeze(0).to(device)).squeeze(0).cpu()  # (C,H,W) CPU

        # FGSM dirigido: restar signo del gradiente en pixel-space (mover hacia target)
        img_adv_pixel = (img_pixel - eps * grad_pixel.sign()).clamp(0.0, 1.0)

        # guardar
        try:
            orig_path = dataset.samples[i][0]
            wnid = Path(orig_path).parent.name
            fname = Path(orig_path).name
        except Exception:
            wnid = str(label_local)
            fname = f"img_{i:06d}.png"

        class_dir = out_dir / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        save_path = class_dir / f"{Path(fname).stem}_fgsm_targeted.png"
        save_image(img_adv_pixel, str(save_path))

        # imprimir progreso
        if (i+1) % print_every == 0 or (i+1) == total:
            print(f"[{i+1}/{total}] guardada -> {save_path}")

        # mostrar primeros ejemplos
        if shown < save_examples_n:
            plot_comparison(img_pixel, img_adv_pixel, save_fig=False, eps=eps)
            shown += 1

    # devolver ImageNet100ValDataset apuntando a out_dir (transform = mismo transform que dataset original)
    labels_json_arg = getattr(dataset, "labels_json", "Labels.json")
    adv_dataset = ImageNet100ValDataset(str(out_dir), transform=dataset.transform, labels_json=labels_json_arg)
    return str(out_dir), adv_dataset

# ---------------------------
# plot comparison (Original | Perturbación | Adversaria)
# ---------------------------
def plot_comparison(img_orig_pixel, img_adv_pixel, save_fig=False, eps=None):
    """
    img_orig_pixel, img_adv_pixel: tensors (C,H,W) en pixel-space [0,1] (CPU tensors)
    """
    img_orig = img_orig_pixel.detach().cpu()
    img_adv = img_adv_pixel.detach().cpu()
    noise = img_adv - img_orig
    # normalizar ruido para visualización
    nmin, nmax = noise.min(), noise.max()
    if (nmax - nmin).abs() < 1e-8:
        noise_vis = torch.zeros_like(noise)
    else:
        noise_vis = (noise - nmin) / (nmax - nmin)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img_orig.permute(1,2,0).numpy())
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Perturbación (visualizada)")
    plt.imshow(noise_vis.permute(1,2,0).numpy())
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Adversarial (FGSM directed)")
    plt.imshow(img_adv.permute(1,2,0).numpy())
    plt.axis("off")

    plt.tight_layout()
    if save_fig:
        eps_str = f"_eps{eps}" if eps is not None else ""
        outp = Path("graficos")
        outp.mkdir(exist_ok=True)
        fname = outp / f"comparativa_fgsm_targeted{eps_str}.png"
        plt.savefig(str(fname), dpi=300, bbox_inches="tight", transparent=True)
        print("Guardado figura:", fname)
    plt.show()

# ---------------------------
# Ejemplo de uso (main)
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)
    preprocess = weights.transforms()

    VAL_DIR = "val.X"
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=preprocess)

    # Elegir WNID objetivo (wombat: ajusta si tu labels.json usa otro wnid)
    wombat_wnid = "n01883070"
    target_idx = wnid_to_model_index(wombat_wnid, weights, labels_json="Labels.json")
    print(f"Target WNID {wombat_wnid} -> índice modelo: {target_idx}")

    eps = 0.05  # ejemplo
    out_dir, adv_ds = generar_dataset_fgsm_dirigido(
        model, val_ds, eps=eps, target_idx=target_idx,
        out_dir=f"FGSM_targeted_{wombat_wnid}", device=device, print_every=100, save_examples_n=2
    )

    print("Generación dirigida finalizada. Adv dataset:", out_dir)
