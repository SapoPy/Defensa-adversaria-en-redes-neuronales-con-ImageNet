
import torch
from ImageNet100ValDataset import ImageNet100ValDataset
from utils import * 

def generar_imagen_RPGD(model, imgs, labels, args):
    eps     = args[0]
    alpha   = args[1]
    iters   = args[2]
    sigma   = args[3]

    imgs, labels, single = ensure_batch(imgs, labels)

    noise = torch.empty_like(imgs).uniform_(-eps, eps)
    adv = imgs + noise

    for _ in range(iters):
        grad = image_gradient(model, adv, labels)
        adv = adv - alpha * grad.sign()

        perturb = torch.clamp(adv - imgs, -eps, eps)
        adv = imgs + perturb

        # ruido aleatorio al final de cada paso
        noise = torch.empty_like(imgs).uniform_(-sigma, sigma)
        adv = adv + noise

    adv = adv.clamp(-3, 3)
    return adv[0] if single else adv


def generar_imagen_RPGD_dirigido(model, imgs, labels, args):
    eps     = args[0]
    alpha   = args[1]
    iters   = args[2]
    sigma   = args[3]

    imgs, labels, single = ensure_batch(imgs, labels)

    noise = torch.empty_like(imgs).uniform_(-eps, eps)
    adv = imgs + noise

    for _ in range(iters):
        grad = image_gradient(model, adv, labels)
        adv = adv + alpha * grad.sign()

        perturb = torch.clamp(adv - imgs, -eps, eps)
        adv = imgs + perturb

        # ruido aleatorio al final de cada paso
        noise = torch.empty_like(imgs).uniform_(-sigma, sigma)
        adv = adv + noise

    adv = adv.clamp(-3, 3)
    return adv[0] if single else adv
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
    sigma = 1/255
    acc_top1, acc_top5 = global_evaluate(model, generar_imagen_RPGD, [eps, alpha, k, sigma], VAL_DIR)
    print(f"Precisión top 1: {acc_top1:.4f}")
    print(f"Precisión top 5: {acc_top5:.4f}")
