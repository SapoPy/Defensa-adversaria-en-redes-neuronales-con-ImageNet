import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from pathlib import Path
from ImageNet100ValDataset import *
from utils import *


def generar_imagen_fgsm(model, img_norm, label, eps):
    """
    img_norm: imagen NORMALIZADA (C,H,W)
    Devuelve adv_norm: imagen adversarial NORMALIZADA
    """
    grad = image_gradient(model, img_norm, label)

    adv_norm = img_norm - eps * grad.sign()

    adv_norm = adv_norm.clamp(-3, 3)

    return adv_norm 


def guardar_dataset_adv(model, dataset, eps, out_dir, device=None, print_every=50):

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)

    for i in range(total):
        img_norm, label_idx = dataset[i]

        adv_norm = generar_imagen_fgsm(model, img_norm, label_idx, eps)

        # Convertir para guardarlo visualmente:
        adv_pixel = denormalize_tensor(adv_norm.unsqueeze(0)).squeeze(0)

        try:
            orig_path = dataset.samples[i][0]
            wnid = Path(orig_path).parent.name
            fname = Path(orig_path).name
        except:
            wnid = str(label_idx)
            fname = f"img_{i:06d}.png"

        class_dir = out_dir / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        save_path = class_dir / f"{Path(fname).stem}_fgsm.png"

        save_image(adv_pixel, str(save_path))

        if (i+1) % print_every == 0 or i+1 == total:
            print(f"[{i+1}/{total}] -> {save_path}")

    return str(out_dir), ImageNet100ValDataset(str(out_dir), transform=dataset.transform)

# ------------------ ejemplo de uso ------------------
if __name__ == "__main__":
    from torchvision.models import resnet34, ResNet34_Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)

    VAL_DIR = "3 clases"
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=transform)

    # eps correcto para ImageNet — el más común
    eps = 0.05

    acc_top1, acc_top5 = global_evaluate(model, generar_imagen_fgsm, eps, VAL_DIR)
    print(f"Precisión top 1: {acc_top1:.4f}")
    print(f"Precisión top 5: {acc_top5:.4f}")
