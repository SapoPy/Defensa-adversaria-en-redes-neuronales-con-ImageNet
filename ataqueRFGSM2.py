import torch
from ImageNet100ValDataset import *
from utils import *
from AtaquesGaussianos2 import *

def generar_imagen_rfgsm(model, imgs, labels, args):
    eps  = args[0]
    alpha = args[1]

    imgs, labels, single = ensure_batch(imgs, labels)

    noise = torch.empty_like(imgs).uniform_(-alpha, alpha)
    grad = image_gradient(model, imgs + noise, labels)

    adv = imgs + noise - eps * grad.sign()
    adv = adv.clamp(-3, 3)

    return adv[0] if single else adv

def generar_imagen_rfgsm_dirigido(model, imgs, labels, args):
    eps  = args[0]
    alpha = args[1]

    imgs, labels, single = ensure_batch(imgs, labels)

    noise = torch.empty_like(imgs).uniform_(-alpha, alpha)
    grad = image_gradient(model, imgs + noise, labels)

    adv = imgs + noise + eps * grad.sign()
    adv = adv.clamp(-3, 3)

    return adv[0] if single else adv


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
