import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from ImageNet100ValDataset import ImageNet100ValDataset
from DirigidoFGSM import wnid_to_model_index

# -------------------------
# Constantes y helpers
# -------------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def denormalize_pixel(tensor_norm, mean=MEAN, std=STD):
    """tensor_norm: (C,H,W) -> pixel-space [0,1]"""
    m = torch.tensor(mean, device=tensor_norm.device).view(-1,1,1)
    s = torch.tensor(std,  device=tensor_norm.device).view(-1,1,1)
    return (tensor_norm * s + m).clamp(0.0, 1.0)

def normalize_from_pixel(tensor_pixel, mean=MEAN, std=STD):
    """tensor_pixel en [0,1] -> normalizado"""
    m = torch.tensor(mean, device=tensor_pixel.device).view(-1,1,1)
    s = torch.tensor(std,  device=tensor_pixel.device).view(-1,1,1)
    return ((tensor_pixel - m) / s)

class AddGaussianNoise:
    """Ruido simple en pixel-space [0,1]"""
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, img_pixel):
        noise = torch.randn_like(img_pixel) * self.std + self.mean
        return (img_pixel + noise).clamp(0.,1.)

def image_gradient_targeted(model, img_norm, target_idx, device): 
    """
    Calcula ∂L/∂x hacia la clase objetivo.
    """
    model = model.to(device).eval()
    x = img_norm.unsqueeze(0).to(device)
    x.requires_grad_(True)

    out = model(x)
    loss = F.cross_entropy(out, torch.tensor([target_idx], device=device))
    model.zero_grad()
    loss.backward()
    grad = x.grad.detach().cpu().squeeze(0)  # (C,H,W)
    return grad


def show_pgd_comparison(img_pixel, adv_pixel, save_fig=False, eps=0.05, out_dir="graficos"):
    """Visualización comparativa"""
    to_show_noise = adv_pixel - img_pixel
    noise_vis = (to_show_noise - to_show_noise.min()) / (to_show_noise.max() - to_show_noise.min() + 1e-8)

    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(img_pixel.permute(1,2,0).numpy()); axs[0].set_title("Original"); axs[0].axis("off")
    axs[1].imshow(noise_vis.permute(1,2,0).numpy()); axs[1].set_title("Perturbación"); axs[1].axis("off")
    axs[2].imshow(adv_pixel.permute(1,2,0).numpy()); axs[2].set_title("Adversaria PGD Dirigida"); axs[2].axis("off")
    plt.suptitle(f"PGD dirigido (ε={eps})")
    plt.tight_layout()
    if save_fig:
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"pgd_dirigido_eps_{eps:.4f}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.show()


# -------------------------
# Ataque PGD dirigido
# -------------------------
def generar_pgd_dirigido(model, val_dataset, target_idx, output_dir="PGD_dirigido",
                          eps=0.05, alpha=0.01, iters=10, seed_std=None,
                          device=None, show_examples_n=3, save_examples=True):
    """
    PGD dirigido:
    - Perturba la imagen en dirección que incrementa la probabilidad de target_idx.
    - Repite el paso varias veces con proyección al rango permitido.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()

    mean_cpu = torch.tensor(MEAN).view(3,1,1)
    std_cpu  = torch.tensor(STD).view(3,1,1)

    if seed_std is None:
        seed_std = eps / 2.0
    noise_adder = AddGaussianNoise(0.0, seed_std)

    os.makedirs(output_dir, exist_ok=True)
    total = len(val_dataset)
    shown = 0
    eps_cpu = torch.tensor(eps).view(1,1,1)

    for idx in range(total):
        img_norm, label_local = val_dataset[idx]
        img_pixel = denormalize_pixel(img_norm.unsqueeze(0)).squeeze(0).cpu()

        # Ruido inicial aleatorio
        img_rand_pixel = noise_adder(img_pixel)
        adv_pixel = img_rand_pixel.clone()

        # Iteraciones PGD
        for _ in range(iters):
            adv_norm = normalize_from_pixel(adv_pixel)
            grad = image_gradient_targeted(model, adv_norm, target_idx, device)

            # Paso dirigido: signo negativo (dirigido)
            adv_norm = adv_norm - (alpha / std_cpu) * grad.sign()

            # Proyección: mantener dentro de eps del original
            orig_norm = normalize_from_pixel(img_pixel)
            delta = torch.clamp(adv_norm - orig_norm, min=-(eps_cpu / std_cpu), max=(eps_cpu / std_cpu))
            adv_norm = orig_norm + delta

            # Clamp a espacio normalizado válido
            min_norm = ((0.0 - mean_cpu) / std_cpu)
            max_norm = ((1.0 - mean_cpu) / std_cpu)
            adv_norm = torch.max(torch.min(adv_norm, max_norm), min_norm)

            # Denormalizar para siguiente iteración
            adv_pixel = denormalize_pixel(adv_norm.unsqueeze(0)).squeeze(0).cpu()

        # Guardar adversaria final
        try:
            wnids = list(val_dataset.class_to_idx.keys())
            wnid = wnids[label_local]
        except Exception:
            wnid = f"class_{label_local}"

        class_dir = os.path.join(output_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, f"img_{idx:06d}_pgd_target_{target_idx}.png")
        save_image(adv_pixel, save_path)

        if (idx+1) % 50 == 0 or (idx+1) == total:
            print(f"[{idx+1}/{total}] guardadas. última: {save_path}")

        if shown < show_examples_n:
            show_pgd_comparison(img_pixel, adv_pixel, save_fig=save_examples, eps=eps)
            shown += 1

    print(f"PGD dirigido completado. Imágenes en: {output_dir}")
    return output_dir


# -------------------------
# Ejemplo de uso
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device).eval()
    transform = weights.transforms()
    val_ds = ImageNet100ValDataset("val.X", transform=transform)

    wombat_wnid = "n01883070"
    TARGET_IDX = wnid_to_model_index(wombat_wnid, weights, labels_json="Labels.json") # 106
    EPS = 0.05
    ALPHA = 0.01
    ITERS = 10
    SEED_STD = EPS / 2.0

    generar_pgd_dirigido(
        model, val_ds, target_idx=TARGET_IDX,
        output_dir=f"PGD_dirigido_target_{TARGET_IDX}",
        eps=EPS, alpha=ALPHA, iters=ITERS,
        seed_std=SEED_STD, device=device,
        show_examples_n=2, save_examples=True
    )
