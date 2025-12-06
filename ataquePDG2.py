import torch
from ImageNet100ValDataset import ImageNet100ValDataset
from utils import * 

def generar_imagen_PGD(model, imgs, labels, args):
    eps     = args[0]
    alpha   = args[1]
    iters   = args[2]

    imgs, labels, single = ensure_batch(imgs, labels)

    # Inicialización aleatoria dentro de [-eps,eps]
    noise = torch.empty_like(imgs).uniform_(-eps, eps)
    adv = imgs + noise

    for _ in range(iters):
        grad = image_gradient(model, adv, labels)
        adv = adv - alpha * grad.sign()

        # Proyecto al L∞ (imagenes normalizadas)
        perturb = torch.clamp(adv - imgs, -eps, eps)
        adv = imgs + perturb

    return adv[0] if single else adv

def generar_imagen_PGD_dirigido(model, imgs, labels, args):
    eps     = args[0]
    alpha   = args[1]
    iters   = args[2]

    imgs, labels, single = ensure_batch(imgs, labels)

    # Inicialización aleatoria dentro de [-eps,eps]
    noise = torch.empty_like(imgs).uniform_(-eps, eps)
    adv = imgs + noise

    for _ in range(iters):
        grad = image_gradient(model, adv, labels)
        adv = adv + alpha * grad.sign()

        # Proyecto al L∞ (imagenes normalizadas)
        perturb = torch.clamp(adv - imgs, -eps, eps)
        adv = imgs + perturb

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

    acc_top1, acc_top5 = global_evaluate(model, generar_imagen_PGD, [eps, alpha, k], VAL_DIR)
    print(f"Precisión top 1: {acc_top1:.4f}")
    print(f"Precisión top 5: {acc_top5:.4f}")
