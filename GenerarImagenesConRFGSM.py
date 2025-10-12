import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from ImageNet100ValDataset import ImageNet100ValDataset  # tu clase


# -------------------------
# Helpers / constantes
# -------------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def denormalize_pixel(tensor_norm, mean=MEAN, std=STD):
    """tensor_norm: (C,H,W) o (B,C,H,W) normalizado -> pixel-space [0,1]"""
    m = torch.tensor(mean, device=tensor_norm.device).view(1, -1, 1, 1) if tensor_norm.dim()==4 else torch.tensor(mean, device=tensor_norm.device).view(-1,1,1)
    s = torch.tensor(std,  device=tensor_norm.device).view(1, -1, 1, 1) if tensor_norm.dim()==4 else torch.tensor(std,  device=tensor_norm.device).view(-1,1,1)
    return (tensor_norm * s + m).clamp(0.0, 1.0)

def normalize_from_pixel(tensor_pixel, mean=MEAN, std=STD):
    """tensor_pixel in [0,1] -> normalized"""
    m = torch.tensor(mean, device=tensor_pixel.device).view(1, -1, 1, 1) if tensor_pixel.dim()==4 else torch.tensor(mean, device=tensor_pixel.device).view(-1,1,1)
    s = torch.tensor(std,  device=tensor_pixel.device).view(1, -1, 1, 1) if tensor_pixel.dim()==4 else torch.tensor(std,  device=tensor_pixel.device).view(-1,1,1)
    return ((tensor_pixel - m) / s)

class AddGaussianNoise:
    """Ruido simple en pixel-space [0,1]"""
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, img_pixel):
        # img_pixel: tensor (C,H,W) en [0,1]
        noise = torch.randn_like(img_pixel) * self.std + self.mean
        return (img_pixel + noise).clamp(0.,1.)

# gradiente simple dL/dx (imagen normalizada)
def image_gradient(model, img_norm, label, device):
    """
    img_norm: tensor (C,H,W) normalized -> se añade batch y se calcula dL/dx (en space normalized).
    devuelve grad (C,H,W) en espacio normalizado en CPU.
    """
    model = model.to(device)
    model_was_train = model.training
    model.eval()

    single = False
    if img_norm.dim() == 3:
        x = img_norm.unsqueeze(0).to(device)
        single = True
    else:
        x = img_norm.to(device)

    x = x.detach()
    x.requires_grad_(True)
    lbl = torch.tensor([label], device=device) if single else label.to(device)

    out = model(x)
    loss = F.cross_entropy(out, lbl)
    model.zero_grad()
    loss.backward()
    grad = x.grad.detach().cpu()  # (1,C,H,W) or (B,C,H,W)

    if single:
        grad = grad.squeeze(0)  # (C,H,W)

    if model_was_train:
        model.train()

    return grad  # en espacio normalized

def show_rfgsm_comparison(img_pixel, adv_pixel, save_fig=False, eps=0.05, out_dir="graficos"):
    """
    img_pixel, adv_pixel: tensors (C,H,W) en pixel-space [0,1], CPU.
    Muestra: original | perturbación (normalizada) | adversaria.
    """
    to_show_noise = adv_pixel - img_pixel
    # normalizar la perturbación para visualizarla (por canal conjunto)
    noise_vis = (to_show_noise - to_show_noise.min()) / (to_show_noise.max() - to_show_noise.min() + 1e-8)

    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(img_pixel.permute(1,2,0).numpy()); axs[0].set_title("Original"); axs[0].axis("off")
    axs[1].imshow(noise_vis.permute(1,2,0).numpy()); axs[1].set_title("Perturbación"); axs[1].axis("off")
    axs[2].imshow(adv_pixel.permute(1,2,0).numpy()); axs[2].set_title("Adversaria"); axs[2].axis("off")
    plt.suptitle(f"R-FGSM (ε={eps})")
    plt.tight_layout()
    if save_fig:
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"rfgsm_eps_{eps:.4f}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.show()


# -------------------------
# Script principal: generar R-FGSM para todo el dataset
# -------------------------
def generar_rfgsm_dataset(model, val_dataset, output_dir="RFGSM_out", eps=0.05, seed_std=None,
                           device=None, show_examples_n=3, save_examples=True):
    """
    Genera R-FGSM para todo `val_dataset` (ImageNet100ValDataset).
    - eps: magnitud en pixel-space.
    - seed_std: std del ruido inicial en pixel-space (si None usa eps/2).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()

    # mean/std en CPU (porque grad y operaciones de guardado se hacen en CPU)
    mean_cpu = torch.tensor(MEAN).view(3,1,1)
    std_cpu  = torch.tensor(STD).view(3,1,1)

    if seed_std is None:
        seed_std = eps / 2.0
    noise_adder = AddGaussianNoise(0.0, seed_std)

    os.makedirs(output_dir, exist_ok=True)

    total = len(val_dataset)
    shown = 0

    # eps en escala pixel (CPU)
    eps_cpu = torch.tensor(eps).view(1,1,1)

    for idx in range(total):
        img_norm, label_local = val_dataset[idx]   # img_norm ya normalizado (C,H,W) en CPU/Tensor
        # denormalizar a pixel-space (CPU)
        img_pixel = denormalize_pixel(img_norm.unsqueeze(0)).squeeze(0).cpu()  # (C,H,W) CPU

        # inicio aleatorio: gaussian en pixel-space
        img_rand_pixel = noise_adder(img_pixel)  # (C,H,W) CPU

        # normalizar para el modelo: pasar a device si hace falta
        img_rand_norm = normalize_from_pixel(img_rand_pixel.unsqueeze(0)).squeeze(0)  # normalized (C,H,W), CPU

        # calcular grad en espacio normalizado (image_gradient devuelve grad en CPU)
        grad_norm = image_gradient(model, img_rand_norm, label_local, device)  # (C,H,W) CPU

        # convertir eps a escala normalizada por canal: eps_norm = eps / std_channel (CPU)
        eps_norm = (eps_cpu / std_cpu)  # broadcast a (3,1,1)

        # aplicar step FGSM en espacio normalizado (todo en CPU)
        adv_norm = img_rand_norm.cpu() + eps_norm * grad_norm.sign()

        # clamp en espacio normalizado: usar mean/std en CPU para límites
        min_norm = ((0.0 - mean_cpu) / std_cpu)
        max_norm = ((1.0 - mean_cpu) / std_cpu)
        adv_norm = torch.max(torch.min(adv_norm, max_norm), min_norm)

        # denormalizar para guardar/visualizar (CPU)
        adv_pixel = denormalize_pixel(adv_norm.unsqueeze(0)).squeeze(0).cpu()

        # Guardar en carpeta por WNID
        try:
            wnids = list(val_dataset.class_to_idx.keys())
            wnid = wnids[label_local]
        except Exception:
            wnid = f"class_{label_local}"

        class_dir = os.path.join(output_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, f"img_{idx:06d}_rfgsm.png")
        save_image(adv_pixel, save_path)

        # progreso
        if (idx+1) % 50 == 0 or (idx+1) == total:
            print(f"[{idx+1}/{total}] guardadas. última: {save_path}")

        # mostrar ejemplos
        if shown < show_examples_n:
            show_rfgsm_comparison(img_pixel, adv_pixel, save_fig=save_examples, eps=eps)
            shown += 1

    print(f"Generación R-FGSM finalizada. Imágenes en: {output_dir}")
    return output_dir



# -------------------------
# run (ejemplo)
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # modelo
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device).eval()

    # dataset: val.X con transform que incluye Normalize
    VAL_DIR = "val.X"
    transform = weights.transforms()  # incluye ToTensor + Normalize
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=transform)

    # parámetros
    OUTPUT_DIR = "RFGSM_out_015"
    EPSILON = 0.15    # magnitud en pixel-space
    SEED_STD = EPSILON / 2.0
    SHOW_N = 1         # mostrar 2 comparativas (original/ruido/adversaria)

    generar_rfgsm_dataset(model, val_ds, output_dir=OUTPUT_DIR, eps=EPSILON,
                          seed_std=SEED_STD, device=device, show_examples_n=SHOW_N, save_examples=True)
