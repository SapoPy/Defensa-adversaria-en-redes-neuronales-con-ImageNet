import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from ImageNet100ValDataset import ImageNet100ValDataset
from DirigidoFGSM import wnid_to_model_index
from DirigidoRFGSM import image_gradient_targeted
from utils import * 

def generar_imagen_PGD(model, img_norm, label, args):
    """
    args[0] = eps      ruido uniforme
    args[1] = alpha    tamaño del paso
    args[2] = iteraciones
    """
    noise = torch.empty_like(img_norm).uniform_(-args[1], args[1])
    img = img_norm + noise
    for _ in range(args[2]):
        grad = image_gradient(model, img, label)
        img = img - args[1] * grad.sign()   
        img = torch.clamp(img - img_norm, min=-args[0], max=args[0])

    return img 

# -------------------------
# Ejemplo de uso
# -------------------------
if __name__ == "__main__":
    from torchvision.models import resnet34, ResNet34_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)

    VAL_DIR = "3 clases"
    val_ds = ImageNet100ValDataset(VAL_DIR, transform=transform)

    # eps correcto para ImageNet — el más común
    eps = 0.05
    alpha = 2/255
    k = 1

    acc_top1, acc_top5 = global_evaluate(model, generar_imagen_PGD, [eps, alpha, k], VAL_DIR)
    print(f"Precisión top 1: {acc_top1:.4f}")
    print(f"Precisión top 5: {acc_top5:.4f}")
