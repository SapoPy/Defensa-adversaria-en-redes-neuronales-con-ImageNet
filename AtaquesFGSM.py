import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from pathlib import Path
from ImageNet100ValDataset import ImageNet100ValDataset

# ImageNet stats
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ------------------ utilidades ------------------
def denormalize_tensor(tensor, mean=MEAN, std=STD):
    """tensor (1,C,H,W) o (C,H,W) normalizado -> pixel-space [0,1]"""
    squeeze = False
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
        squeeze = True
    m = torch.tensor(mean, device=tensor.device).view(-1,1,1)
    s = torch.tensor(std,  device=tensor.device).view(-1,1,1)
    out = (tensor * s + m).clamp(0.0, 1.0)
    return out

def normalize_for_model(tensor, mean=MEAN, std=STD):
    m = torch.tensor(mean, device=tensor.device).view(-1,1,1)
    s = torch.tensor(std,  device=tensor.device).view(-1,1,1)
    return (tensor - m) / s

def image_gradient(model, img, label, device=None, to_pixel_space=False):
    """
    img: tensor normalizado (C,H,W) que entrarías al modelo.
    label: int
    devuelve grad (C,H,W) en CPU; si to_pixel_space=True, grad se escala por 1/std.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()

    x = img.unsqueeze(0).to(device).detach()
    x.requires_grad_(True)
    y = torch.tensor([label], device=device)

    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    grad = x.grad.detach().cpu().squeeze(0)   # (C,H,W) en espacio normalizado
    if to_pixel_space:
        std_t = torch.tensor(STD).view(-1,1,1)
        grad = grad / std_t
    return grad

# ------------------ generar dataset FGSM (imprime por imagen) ------------------
def generar_dataset_fgsm_print(model, dataset, eps, out_dir, device=None, print_every=1):
    """
    Recorre `dataset` (ImageNet100ValDataset), genera FGSM (sumando signo del grad en pixel-space)
    y guarda cada imagen perturbada en out_dir/wnid/.
    Imprime por imagen: "Procesadas i/N".
    Devuelve (out_dir_str, ImageNet100ValDataset(out_dir, transform=dataset.transform))
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    # iterar por índice para poder mapear a samples y obtener wnid
    for i in range(total):
        img_norm, label_idx = dataset[i]         # img_norm: NORMALIZADO (C,H,W)
        # 1) grad en pixel-space
        grad_pixel = image_gradient(model, img_norm, label_idx, device=device, to_pixel_space=True)  # CPU (C,H,W)

        # 2) denormalizar la imagen original a pixel-space
        img_pixel = denormalize_tensor(img_norm.unsqueeze(0).to(device)).squeeze(0).cpu()            # CPU (C,H,W)

        # 3) FGSM: sumar signo del grad (pixel-space)
        img_adv_pixel = (img_pixel + eps * grad_pixel.sign()).clamp(0.0, 1.0)  # CPU (C,H,W)

        # 4) guardar (manteniendo estructura wnid)
        try:
            orig_path = dataset.samples[i][0]
            wnid = Path(orig_path).parent.name
            fname = Path(orig_path).name
        except Exception:
            wnid = str(label_idx)
            fname = f"img_{i:06d}.png"

        class_dir = out_dir / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        save_path = class_dir / f"{Path(fname).stem}_fgsm.png"
        save_image(img_adv_pixel, str(save_path))

        # imprimir progreso (por imagen)
        if (i+1) % print_every == 0 or (i+1) == total:
            print(f"Procesadas {i+1}/{total} imágenes -> guardado en: {save_path}")

    # devolver un dataset apuntando a la carpeta con transform original
    adv_dataset = ImageNet100ValDataset(str(out_dir), transform=dataset.transform)
    return str(out_dir), adv_dataset

def plot_comparison(img_orig_pixel, img_adv_pixel, save_fig=False, eps=None):
    """
    Muestra una comparación estilo 'show_noise_example':
    - Original
    - Ruido (normalizado a [0,1] solo para visualizar)
    - Imagen resultante (adversarial)
    
    Parámetros:
        img_orig_pixel : tensor (C,H,W) en pixel-space [0,1]
        img_adv_pixel  : tensor (C,H,W) en pixel-space [0,1]
        save_fig       : bool, si True guarda el gráfico
        eps            : opcional, valor de epsilon para incluir en el título del archivo
    """
    # asegurar que estén en CPU y tensor
    img_orig_pixel = img_orig_pixel.detach().cpu()
    img_adv_pixel = img_adv_pixel.detach().cpu()

    # calcular ruido
    noise = img_adv_pixel - img_orig_pixel
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img_orig_pixel.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Perturbación añadida")
    plt.imshow(noise.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Imagen adversarial")
    plt.imshow(img_adv_pixel.permute(1, 2, 0))
    plt.axis("off")

    plt.tight_layout()

    if save_fig:
        eps_str = f" eps={eps}" if eps is not None else ""
        file_name = f"graficos/comparativa_FGSM{eps_str}.png"
        plt.savefig(file_name, dpi=300, transparent=True, bbox_inches="tight")
        print(f"Gráfico guardado en {file_name}")

    plt.show()

# ------------------ ejemplo de uso ------------------
if __name__ == "__main__":
    from torchvision.models import resnet34, ResNet34_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)
    preprocess = weights.transforms()

    VAL_DIR = "val.X"   # tu carpeta
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=preprocess)

    # Generar dataset FGSM (imprime por imagen)
    eps = 0.15
    out_dir, adv_ds = generar_dataset_fgsm_print(model, val_ds, eps, out_dir="FGSM_out_015", device=device, print_every=1)

    # Mostrar comparación para la primera imagen (ejemplo)
    img_norm, _ = val_ds[0]                                 # NORMALIZADO (C,H,W)
    img_pixel = denormalize_tensor(img_norm.unsqueeze(0).to(device)).squeeze(0).cpu()
    adv_path = list(Path(out_dir).glob(f"{list(val_ds.class_to_idx.keys())[0]}/*fgsm.png"))[0]
    img_adv = ImageNet100ValDataset  # fallback (no usar)
    # mejor: leer el adv guardado
    from PIL import Image
    adv_img = Image.open(adv_path).convert("RGB")
    adv_tensor = torch.tensor(preprocess.transforms[2](adv_img)) if hasattr(preprocess, "transforms") else None
    # Si quieres mostrar directamente desde tensors: usa plot_comparison(img_pixel, img_adv_pixel_tensor)
