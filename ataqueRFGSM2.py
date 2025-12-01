import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from pathlib import Path
from ImageNet100ValDataset import *
from utils import *
from AtaquesGaussianos2 import *

def generar_imagen_rfgsm(model, img_norm, label, args):
    """
    args[0] = eps
    args[1] 0 alpha
    img_norm: imagen NORMALIZADA (C,H,W)
    Devuelve adv_norm: imagen adversarial NORMALIZADA
    """
    noise = torch.empty_like(img_norm).uniform_(-args[1], args[1])
    grad = image_gradient(model, img_norm + noise, label)

    # FGSM en espacio NORMALIZADO
    adv_norm = img_norm + noise - args[0] * grad.sign()

    # Clamp en normalizado evita explosiones
    adv_norm = adv_norm.clamp(-3, 3)

    return adv_norm 


# ------------------ ejemplo de uso ------------------
if __name__ == "__main__":
    from torchvision.models import resnet34, ResNet34_Weights
    from torchvision.models import resnet50, ResNet50_Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)

    VAL_DIR = "3 clases"
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=transform)

    # eps correcto para ImageNet — el más común
    eps = 0.05
    alpha = 2/255

    acc_top1, acc_top5 = global_evaluate(model, generar_imagen_rfgsm, [eps, alpha], VAL_DIR)
    print(f"Precisión top 1: {acc_top1:.4f}")
    print(f"Precisión top 5: {acc_top5:.4f}")
